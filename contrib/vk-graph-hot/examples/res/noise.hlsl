const uint k = 1103515245u;

float3 hash(uint3 x) {
    x = ((x >> 8u) ^ x.yzx) * k;
    x = ((x >> 8u) ^ x.yzx) * k;
    x = ((x >> 8u) ^ x.yzx) * k;

    return float3(x) * (1.0 / float(0xffffffffu));
}
