//! TODO

#![warn(missing_docs)]

pub use vk_graph::{
    BindGraph, Bound, ClearColorValue, Graph,
    cmd::{
        BuildAccelerationStructureIndirectInfo, BuildAccelerationStructureInfo, CommandRef, LoadOp,
        PipelineCommandRef, StoreOp, UpdateAccelerationStructureIndirectInfo,
        UpdateAccelerationStructureInfo,
    },
    driver::{
        DriverError,
        accel_struct::{
            AccelerationStructure, AccelerationStructureGeometry,
            AccelerationStructureGeometryData, AccelerationStructureGeometryInfo,
            AccelerationStructureInfo, AccelerationStructureInfoBuilder, AccelerationStructureSize,
            DeviceOrHostAddress,
        },
        ash::vk,
        buffer::{Buffer, BufferInfo, BufferInfoBuilder, BufferSubresourceRange},
        cmd_buf::CommandBuffer,
        compute::{ComputePipeline, ComputePipelineInfo, ComputePipelineInfoBuilder},
        device::{Device, DeviceInfo, DeviceInfoBuilder},
        graphic::{
            BlendInfo, BlendInfoBuilder, DepthStencilInfo, DepthStencilInfoBuilder,
            GraphicPipeline, GraphicPipelineInfo, GraphicPipelineInfoBuilder, StencilMode,
        },
        image::{
            Image, ImageInfo, ImageInfoBuilder, ImageViewInfo, ImageViewInfoBuilder, SampleCount,
        },
        instance::{Instance, InstanceInfo, InstanceInfoBuilder},
        physical_device::{
            AccelerationStructureProperties, PhysicalDevice, RayQueryFeatures, RayTraceFeatures,
            RayTraceProperties, Vulkan10Features, Vulkan10Limits, Vulkan10Properties,
            Vulkan11Features, Vulkan11Properties, Vulkan12Features, Vulkan12Properties,
        },
        ray_trace::{
            RayTracePipeline, RayTracePipelineInfo, RayTracePipelineInfoBuilder,
            RayTraceShaderGroup, RayTraceShaderGroupType,
        },
        render_pass::ResolveMode,
        shader::{SamplerInfo, SamplerInfoBuilder, Shader, ShaderBuilder, SpecializationMap},
        surface::Surface,
        swapchain::{
            Swapchain, SwapchainError, SwapchainImage, SwapchainInfo, SwapchainInfoBuilder,
        },
        sync::AccessType,
    },
    node::{
        AccelerationStructureLeaseNode, AccelerationStructureNode, AnyAccelerationStructureNode,
        AnyBufferNode, AnyImageNode, BufferLeaseNode, BufferNode, ImageLeaseNode, ImageNode,
        SwapchainImageNode,
    },
    pool::{
        Lease, Pool, PoolInfo, PoolInfoBuilder,
        alias::{Alias, AliasWrapper},
        fifo::FifoPool,
        hash::HashPool,
        lazy::LazyPool,
    },
};

#[allow(deprecated)]
pub use vk_graph::{
    driver::graphic::{BlendMode, DepthStencilMode},
    pool::alias::AliasPool,
};
