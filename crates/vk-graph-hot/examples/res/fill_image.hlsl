#include "plasma.hlsl"

struct PushConst {
    uint frame_index;
    uint frame_width;
    uint frame_height;
};

[[vk::push_constant]]
cbuffer PushConst {
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
    float2 extent = float2(1.0, float(push_const.frame_width) / float(push_const.frame_height));
    float2 offset = vertex.tex_coord * float(push_const.frame_width);
    float4 color = plasma(
        offset,
        extent,
        float(push_const.frame_index) / 12440.0,
        0.215,
        1.0,
        0.143,
        float3(0.6, 0.3, 0.2)
    );

    return color;
}
