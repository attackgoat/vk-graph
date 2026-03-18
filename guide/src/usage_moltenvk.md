# MoltenVK

Vulkan is emulated on Apple platforms using MoltenVK.

> [!WARNING]
> MoltenVK does not support all Vulkan features and has limited extension and format support. Pay
> particular attention to these areas:
> - Bindless descriptor count limit
> - Hardware queues provided for execution
> - Indirect drawing command support
> - Image format support

Support for MoltenVK is best-effort and may not always be up to date. In the event that any
`vk-graph` workflow does not work using MoltenVK please
[_open an issue_](https://github.com/attackgoat/vk-graph/issues) <i class="fa-solid fa-arrow-up-right-from-square"></i>.
