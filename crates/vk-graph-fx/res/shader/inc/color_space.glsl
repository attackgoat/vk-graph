float linear_to_srgb(float val)
{
    if (val <= 0.0031308) {
        return val * 12.92;
    } else {
        return pow(val, (1.0 / 2.4)) * 1.055 - 0.055;
    }
}

vec3 linear_to_srgb(vec3 val)
{
    return vec3(
        linear_to_srgb(val.x),
        linear_to_srgb(val.y),
        linear_to_srgb(val.z)
    );
}