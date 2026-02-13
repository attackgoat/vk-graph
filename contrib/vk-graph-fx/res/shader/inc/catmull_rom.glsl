//adapted from: https://www.shadertoy.com/view/MtVGWz
//original: https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
//note: see also http://www.decarpentier.nl/2d-catmull-rom-in-4-samples.

/*
vec4 sampleLevel0( vec2 uv )
{
    return texture( iChannel0, uv, -10.0 );
}
*/

// note: entirely stolen from https://gist.github.com/TheRealMJP/c83b8c0f46b63f3a88a5986f4fa982b1
//
// Samples a texture with Catmull-Rom filtering, using 9 texture fetches instead of 16.
// See http://vec3.ca/bicubic-filtering-in-fewer-taps/ for more details
vec4 sample_catmull_rom(sampler2D image, vec2 uv)
{
    vec2 image_size = vec2(textureSize(image, 0));

    // We're going to sample a a 4x4 grid of texels surrounding the target UV coordinate. We'll do this by rounding
    // down the sample location to get the exact center of our "starting" texel. The starting texel will be at
    // location [1, 1] in the grid, where [0, 0] is the top left corner.
    vec2 samplePos = uv * image_size;
    vec2 texPos1 = floor(samplePos - 0.5) + 0.5;

    // Compute the fractional offset from our starting texel to our original sample location, which we'll
    // feed into the Catmull-Rom spline function to get our filter weights.
    vec2 f = samplePos - texPos1;

    // Compute the Catmull-Rom weights using the fractional offset that we calculated earlier.
    // These equations are pre-expanded based on our knowledge of where the texels will be located,
    // which lets us avoid having to evaluate a piece-wise function.
    vec2 w0 = f * ( -0.5 + f * (1.0 - 0.5*f));
    vec2 w1 = 1.0 + f * f * (-2.5 + 1.5*f);
    vec2 w2 = f * ( 0.5 + f * (2.0 - 1.5*f) );
    vec2 w3 = f * f * (-0.5 + 0.5 * f);
    
    // Work out weighting factors and sampling offsets that will let us use bilinear filtering to
    // simultaneously evaluate the middle 2 samples from the 4x4 grid.
    vec2 w12 = w1 + w2;
    vec2 offset12 = w2 / w12;

    // Compute the final UV coordinates we'll use for sampling the texture
    vec2 texPos0 = texPos1 - vec2(1.0);
    vec2 texPos3 = texPos1 + vec2(2.0);
    vec2 texPos12 = texPos1 + offset12;

    texPos0 /= image_size;
    texPos3 /= image_size;
    texPos12 /= image_size;

    vec4 result = vec4(0.0);
    result += texture( image, vec2(texPos0.x,  texPos0.y)) * w0.x * w0.y;
    result += texture( image, vec2(texPos12.x, texPos0.y)) * w12.x * w0.y;
    result += texture( image, vec2(texPos3.x,  texPos0.y)) * w3.x * w0.y;

    result += texture( image, vec2(texPos0.x,  texPos12.y)) * w0.x * w12.y;
    result += texture( image, vec2(texPos12.x, texPos12.y)) * w12.x * w12.y;
    result += texture( image, vec2(texPos3.x,  texPos12.y)) * w3.x * w12.y;

    result += texture( image, vec2(texPos0.x,  texPos3.y)) * w0.x * w3.y;
    result += texture( image, vec2(texPos12.x, texPos3.y)) * w12.x * w3.y;
    result += texture( image, vec2(texPos3.x,  texPos3.y)) * w3.x * w3.y;

    return result;
}

/*
//note: uniform pdf rand [0;1[
vec3 hash32n(vec2 p)
{
	p  = fract(p * vec2(5.3987, 5.4421));
    p += dot(p.yx, p.xy +  vec2(21.5351, 14.3137));
	return fract(vec3(p.x * p.y * 95.4307, p.x * p.y * 97.5901, p.x * p.y * 93.8369));
}

void mainImage( out vec4 fragColor, in vec2 fragCoord )
{
	vec2 uv = fragCoord.xy / iResolution.xy;
    
    vec2 sample_uv = uv;
    sample_uv.x = mod( sample_uv.x, 0.5 );
   	sample_uv += 0.01 * vec2( cos(iTime), sin(iTime) );
    sample_uv *= (iMouse.z>0.0)? iMouse.x * 0.001 : 0.06125;
    
    vec4 smpl;
    if ( uv.x < 0.5 )
    	smpl = SampleTextureCatmullRom( sample_uv, iChannelResolution[0].xy );
    else
        smpl = sampleLevel0( sample_uv );

    fragColor = smpl;
    fragColor -= step(abs(uv.x-0.5), 0.001);
    fragColor.rgb += (hash32n(uv+fract(iTime))+hash32n(uv+0.1337*fract(iTime))-1.0)/255.0; //dither output
}
*/