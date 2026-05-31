#version 460 core

layout(set = 0, binding = 0) uniform sampler2D image_sampler_llb;

layout(location = 0) in vec2 input_texcoord;
layout(location = 1) in vec4 input_color;

layout(location = 0) out vec4 output_color;

void main()
{
    output_color = input_color * texture(image_sampler_llb, input_texcoord);
}
