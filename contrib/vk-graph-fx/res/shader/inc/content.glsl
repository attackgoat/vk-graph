/// Content aware scaling in "fill" mode where a piece of content is scaled such that all four sides
/// are covered by the destination frame and there is no whitespace or void leftover.
///
/// Arguments:
///  src_size: The size of something
///  dst_size: The size of the place you want to put something
///  dst_coord: The pixel coordinate you want to get the source UV of
///
/// Returns:
///  vec2: The normalized UV coordiate of the source image that you should sample
///        in order to fill the dst_coord pixel.
vec2 content_fill(uvec2 src_size, uvec2 dst_size, uvec2 dst_coord)
{
    vec2 src = vec2(src_size);
    vec2 dst = vec2(dst_size);

    // Scale brings the source image into a "fill" size (as opposed to "fit")
    float scale = max(dst.x / src.x, dst.y / src.y);

    // Offset centers the source image in the destination viewport
    vec2 offset = (src * scale - dst) * 0.5;

    // The destination x/y coordinates are now mappable to source pixels
    return (vec2(dst_coord) + offset) / scale / src;
}