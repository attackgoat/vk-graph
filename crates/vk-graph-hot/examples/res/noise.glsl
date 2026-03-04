const uint k = 1103515245u;

vec3 hash(uvec3 x) {
    x = ((x >> 8u) ^ x.yzx) * k;
    x = ((x >> 8u) ^ x.yzx) * k;
    x = ((x >> 8u) ^ x.yzx) * k;

    return vec3(x) * (1.0 / float(0xffffffffu));
}
