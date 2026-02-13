#version 450

layout(location = 0) in vec2 i_pos;
layout(location = 1) in vec2 i_uv;
layout(location = 2) in uint i_color;

layout(location = 0) out vec4 o_color;
layout(location = 1) out vec2 o_uv;

layout(push_constant) uniform PushConstants{
    vec2 screen_size;
};


vec4 linear_from_srgba(vec4 srgba)
{
    bvec4 cutoff = lessThan(srgba, vec4(0.04045));
    vec4 higher = pow((srgba + vec4(0.055))/vec4(1.055), vec4(2.4));
    vec4 lower = srgba/vec4(12.92);

    return mix(higher, lower, cutoff);
}

void main(){
    gl_Position = vec4(
            2.0 * i_pos.x / screen_size.x - 1.0,
            2.0 * i_pos.y / screen_size.y - 1.0,
            0.0, 1.0);
    o_color = unpackUnorm4x8(i_color);
    o_color = linear_from_srgba(o_color);
    o_uv = i_uv;
}
