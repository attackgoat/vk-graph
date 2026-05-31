#version 460 core

layout(constant_id = 0) const uint NUM_PAGES = 1;

layout(push_constant) uniform PushConstants {
    layout(offset = 80) vec4 glyph_color;
    layout(offset = 96) vec4 outline_color;
} push_constants;

layout(set = 0, binding = 0) uniform sampler2D pages_sampler_nnr[NUM_PAGES];

layout(location = 0) in vec2 texcoord;
layout(location = 1) in flat int page_in;

layout(location = 0) out vec4 color;

void main() {
    vec2 page_colors = texture(pages_sampler_nnr[page_in], texcoord).rg;
    color = page_colors.r * push_constants.glyph_color
          + page_colors.g * push_constants.outline_color;
}