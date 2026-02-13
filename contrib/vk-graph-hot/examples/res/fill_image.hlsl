#include "noise.hlsl"

struct PushConst {
    uint frame_index;
    uint frame_width;
    uint frame_height;
};

[[vk::push_constant]]
cbuffer {
    PushConst push_const;
};

struct Vertex {
    float4 position: SV_POSITION;
    [[vk::location(0)]] float2 tex_coord: TEXCOORD0;
};

Vertex vertex_main(uint vertex_id: SV_VERTEXID) {
    Vertex vertex;

    vertex.tex_coord = float2((vertex_id << 1) & 2, vertex_id & 2);
    vertex.position = float4(vertex.tex_coord * float2(2, -2) + float2(-1, 1), 0, 1);

    return vertex;
}

float4 fragment_main(Vertex vertex): SV_TARGET {
    uint3 coord;
    coord.x = uint(vertex.tex_coord.x * float(push_const.frame_width));
    coord.y = uint(vertex.tex_coord.y * float(push_const.frame_height));
    coord.z = push_const.frame_index;

    return float4(hash(coord), 1.0);
}
