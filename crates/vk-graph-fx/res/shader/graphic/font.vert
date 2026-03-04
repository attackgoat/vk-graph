#version 460 core

layout(constant_id = 0) const uint NUM_PAGES = 1;

layout(push_constant) uniform PushConstants {
    layout(offset = 0) mat4 view_proj;
    layout(offset = 64) vec2 framebuffer_extent_inverse;
} push_constants;

layout(set = 0, binding = 0) uniform sampler2D pages_sampler_nnr[NUM_PAGES];

layout(location = 0) in vec2 position_in;
layout(location = 1) in vec2 texcoord_in;
layout(location = 2) in int page_in;

layout(location = 0) out vec2 texcoord_out;
layout(location = 1) out int page_out;

void main() {
    texcoord_out = texcoord_in / textureSize(pages_sampler_nnr[page_in], 0);
    page_out = page_in;

    gl_Position = push_constants.view_proj
        * vec4(position_in * push_constants.framebuffer_extent_inverse, 0, 1);
}