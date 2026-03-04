#version 450

layout(location = 0) in vec4 i_color;
layout(location = 1) in vec2 i_uv;

layout(location = 0) out vec4 o_color;

layout(binding = 0, set = 0) uniform sampler2D font_sampler_lle;

vec4 srgba_from_linear(vec4 linear){
    bvec4 cutoff = lessThan(linear, vec4(0.0031308));
    vec4 higher = vec4(1.055)*pow(linear, vec4(1.0/2.4)) - vec4(0.055);
    vec4 lower = linear * vec4(12.92);

    return mix(higher, lower, cutoff);
}

void main(){
    o_color = srgba_from_linear(i_color * texture(font_sampler_lle, i_uv));
}
