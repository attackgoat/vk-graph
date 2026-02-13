#version 460 core

layout(location = 0) in vec2 input_position;
layout(location = 1) in vec2 input_texcoord;
layout(location = 2) in uint input_color;

layout(push_constant) uniform PushConstant {
    vec2 g_dims_rcp;
};

out gl_PerVertex {
    vec4 gl_Position;
};

layout(location = 0) out vec2 output_texcoord;
layout(location = 1) out vec4 output_color;

void main() {
    output_color = unpackUnorm4x8(input_color);
    output_texcoord = input_texcoord;

    gl_Position = vec4(input_position * g_dims_rcp * 2.0 - 1.0, 0, 1);
}
