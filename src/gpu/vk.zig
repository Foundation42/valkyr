//! Headless Vulkan compute context.
//!
//! Everything a compute-only program needs and nothing more: instance,
//! physical device pick (prefer discrete), logical device with one
//! compute-capable queue, and a command pool. No surface, no swapchain,
//! no GLFW — this isn't a renderer, it's a kernel host.
//!
//! Validation layers are enabled in Debug / ReleaseSafe builds so that
//! synchronisation mistakes surface as VK_ERROR_VALIDATION_FAILED_EXT
//! instead of silent corruption. They're off in ReleaseFast — the layer
//! overhead is real (10-30%) and the kernels we ship should be correct
//! by then.

const std = @import("std");
const builtin = @import("builtin");

pub const c = @cImport({
    @cInclude("vulkan/vulkan.h");
});

pub fn check(result: c.VkResult) !void {
    if (result == c.VK_SUCCESS) return;
    std.debug.print("Vulkan call failed: VkResult={d}\n", .{result});
    return error.VkFailed;
}

/// Returns true iff `name` appears in vkEnumerateInstanceLayerProperties.
/// Used so we only enable the validation layer when the SDK package is
/// actually installed — otherwise vkCreateInstance fails outright with
/// VK_ERROR_LAYER_NOT_PRESENT, which is a worse failure mode than just
/// running without validation.
fn hasInstanceLayer(name: []const u8) bool {
    var count: u32 = 0;
    if (c.vkEnumerateInstanceLayerProperties(&count, null) != c.VK_SUCCESS) return false;
    if (count == 0) return false;
    var props: [32]c.VkLayerProperties = undefined;
    var got: u32 = @min(count, @as(u32, props.len));
    if (c.vkEnumerateInstanceLayerProperties(&got, &props) != c.VK_SUCCESS) return false;
    for (props[0..got]) |lp| {
        const layer_name = std.mem.sliceTo(@as([*:0]const u8, @ptrCast(&lp.layerName)), 0);
        if (std.mem.eql(u8, layer_name, name)) return true;
    }
    return false;
}

fn makeApiVersion(variant: u32, major: u32, minor: u32, patch: u32) u32 {
    return (variant << 29) | (major << 22) | (minor << 12) | patch;
}

const enable_validation = switch (builtin.mode) {
    .Debug, .ReleaseSafe => true,
    .ReleaseFast, .ReleaseSmall => false,
};

pub const Context = struct {
    instance: c.VkInstance,
    physical_device: c.VkPhysicalDevice,
    device: c.VkDevice,
    /// Index of the queue family we're using. Cached for command-pool
    /// creation and for any future per-family resource (e.g. timeline
    /// semaphores scoped to the same family).
    queue_family: u32,
    queue: c.VkQueue,
    cmd_pool: c.VkCommandPool,
    /// Cached so callers can size staging buffers and pick alignments
    /// without re-querying every time.
    props: c.VkPhysicalDeviceProperties,

    // Ownership flags so the same struct works for both modes:
    //   .init()   — valkyr creates everything; deinit destroys it all.
    //   .attach() — host engine owns instance/device/cmd_pool; deinit
    //               must NOT touch them. (queue + physical_device are
    //               handles, not destroyable resources, so no flag.)
    // The reason this matters: in the embedded use case (Matryoshka
    // hosting valkyr) the host's render loop has already set up a
    // device + queue + cmd_pool that we share. Destroying any of them
    // from valkyr's deinit would crash the host on next frame.
    owns_instance: bool,
    owns_device: bool,
    owns_cmd_pool: bool,

    pub fn init(allocator: std.mem.Allocator) !Context {
        _ = allocator; // reserved — extension/layer enumeration may need it

        // ── Instance ────────────────────────────────────────────────
        var app_info = std.mem.zeroes(c.VkApplicationInfo);
        app_info.sType = c.VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "tripvulkan";
        app_info.applicationVersion = makeApiVersion(0, 0, 1, 0);
        app_info.pEngineName = "tripvulkan";
        app_info.engineVersion = makeApiVersion(0, 0, 1, 0);
        app_info.apiVersion = makeApiVersion(0, 1, 3, 0);

        const validation_layers = [_][*:0]const u8{"VK_LAYER_KHRONOS_validation"};
        const want_validation = enable_validation and hasInstanceLayer("VK_LAYER_KHRONOS_validation");

        var ici = std.mem.zeroes(c.VkInstanceCreateInfo);
        ici.sType = c.VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        ici.pApplicationInfo = &app_info;
        if (want_validation) {
            ici.enabledLayerCount = validation_layers.len;
            ici.ppEnabledLayerNames = @ptrCast(&validation_layers);
        } else if (enable_validation) {
            std.debug.print(
                "note: VK_LAYER_KHRONOS_validation not installed; running without validation. " ++
                    "Install vulkan-validation-layers (Arch) / vulkan-validationlayers (Debian) to enable.\n",
                .{},
            );
        }

        var instance: c.VkInstance = null;
        try check(c.vkCreateInstance(&ici, null, &instance));
        errdefer c.vkDestroyInstance(instance, null);

        // ── Physical device pick ────────────────────────────────────
        // Prefer DISCRETE_GPU. Fall back to whatever's first if no
        // discrete device is present (e.g. CI box, integrated-only laptop).
        var dev_count: u32 = 0;
        try check(c.vkEnumeratePhysicalDevices(instance, &dev_count, null));
        if (dev_count == 0) return error.NoVulkanDevice;
        var devs: [16]c.VkPhysicalDevice = undefined;
        const cap = @min(dev_count, devs.len);
        dev_count = cap;
        try check(c.vkEnumeratePhysicalDevices(instance, &dev_count, &devs));

        var picked: c.VkPhysicalDevice = devs[0];
        var picked_props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(picked, &picked_props);
        for (devs[0..dev_count]) |pd| {
            var p: c.VkPhysicalDeviceProperties = undefined;
            c.vkGetPhysicalDeviceProperties(pd, &p);
            if (p.deviceType == c.VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
                picked = pd;
                picked_props = p;
                break;
            }
        }

        // ── Queue family pick ───────────────────────────────────────
        // Any queue family with COMPUTE_BIT works. Prefer one that's
        // compute-only over graphics+compute when available — async-
        // compute queues sometimes carry less front-end contention. If
        // none, fall back to the universal queue family (which always
        // exists on a graphics-capable device).
        var qf_count: u32 = 0;
        c.vkGetPhysicalDeviceQueueFamilyProperties(picked, &qf_count, null);
        var qfs: [16]c.VkQueueFamilyProperties = undefined;
        const qf_cap = @min(qf_count, qfs.len);
        qf_count = qf_cap;
        c.vkGetPhysicalDeviceQueueFamilyProperties(picked, &qf_count, &qfs);

        var queue_family: ?u32 = null;
        // First pass: dedicated compute (compute set, graphics not set).
        for (qfs[0..qf_count], 0..) |qf, i| {
            const has_compute = (qf.queueFlags & c.VK_QUEUE_COMPUTE_BIT) != 0;
            const has_graphics = (qf.queueFlags & c.VK_QUEUE_GRAPHICS_BIT) != 0;
            if (has_compute and !has_graphics) {
                queue_family = @intCast(i);
                break;
            }
        }
        // Second pass: any compute-capable family.
        if (queue_family == null) {
            for (qfs[0..qf_count], 0..) |qf, i| {
                if ((qf.queueFlags & c.VK_QUEUE_COMPUTE_BIT) != 0) {
                    queue_family = @intCast(i);
                    break;
                }
            }
        }
        const qf_index = queue_family orelse return error.NoComputeQueue;

        // ── Logical device + queue ──────────────────────────────────
        const queue_priority: f32 = 1.0;
        var dqci = std.mem.zeroes(c.VkDeviceQueueCreateInfo);
        dqci.sType = c.VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        dqci.queueFamilyIndex = qf_index;
        dqci.queueCount = 1;
        dqci.pQueuePriorities = &queue_priority;

        var dci = std.mem.zeroes(c.VkDeviceCreateInfo);
        dci.sType = c.VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &dqci;

        var device: c.VkDevice = null;
        try check(c.vkCreateDevice(picked, &dci, null, &device));
        errdefer c.vkDestroyDevice(device, null);

        var queue: c.VkQueue = null;
        c.vkGetDeviceQueue(device, qf_index, 0, &queue);

        // ── Command pool ────────────────────────────────────────────
        // RESET_COMMAND_BUFFER_BIT lets us reuse individual buffers
        // rather than reset the whole pool — saves a syscall per
        // dispatch when we start chaining kernels per token.
        var cpci = std.mem.zeroes(c.VkCommandPoolCreateInfo);
        cpci.sType = c.VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        cpci.flags = c.VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpci.queueFamilyIndex = qf_index;

        var cmd_pool: c.VkCommandPool = null;
        try check(c.vkCreateCommandPool(device, &cpci, null, &cmd_pool));

        return .{
            .instance = instance,
            .physical_device = picked,
            .device = device,
            .queue_family = qf_index,
            .queue = queue,
            .cmd_pool = cmd_pool,
            .props = picked_props,
            .owns_instance = true,
            .owns_device = true,
            .owns_cmd_pool = true,
        };
    }

    /// Embedded-mode constructor: borrow Vulkan handles from a host
    /// (e.g. a game engine that already created its own device + queue +
    /// command pool). Valkyr will record dispatches into the host's
    /// world without destroying any of the host's state at deinit.
    ///
    /// Caller MUST guarantee the handles outlive this Context. The
    /// `cmd_pool` must have been created with the same `queue_family`
    /// the queue belongs to, and with `RESET_COMMAND_BUFFER_BIT` so the
    /// recorder can recycle individual buffers per forward pass.
    ///
    /// The physical-device props are re-queried here rather than passed
    /// in — `vkGetPhysicalDeviceProperties` is a cheap driver-side call
    /// and saves the host caller from filling a `VkPhysicalDeviceProperties`
    /// they probably don't keep around.
    pub fn attach(
        instance: c.VkInstance,
        physical_device: c.VkPhysicalDevice,
        device: c.VkDevice,
        queue: c.VkQueue,
        queue_family: u32,
        cmd_pool: c.VkCommandPool,
    ) Context {
        var props: c.VkPhysicalDeviceProperties = undefined;
        c.vkGetPhysicalDeviceProperties(physical_device, &props);
        return .{
            .instance = instance,
            .physical_device = physical_device,
            .device = device,
            .queue_family = queue_family,
            .queue = queue,
            .cmd_pool = cmd_pool,
            .props = props,
            .owns_instance = false,
            .owns_device = false,
            .owns_cmd_pool = false,
        };
    }

    pub fn deinit(self: *Context) void {
        if (self.owns_cmd_pool) c.vkDestroyCommandPool(self.device, self.cmd_pool, null);
        if (self.owns_device) c.vkDestroyDevice(self.device, null);
        if (self.owns_instance) c.vkDestroyInstance(self.instance, null);
    }

    /// Human-readable device name (null-terminated, owned by Vulkan
    /// driver — do not free). Useful for the "GPU OK on <NAME>" line.
    pub fn deviceName(self: *const Context) [*:0]const u8 {
        return @ptrCast(&self.props.deviceName);
    }
};
