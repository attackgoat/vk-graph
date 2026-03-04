const float U[6] = {0, 0, 1, 1, 1, 0};
const float V[6] = {0, 1, 0, 1, 0, 1};
const float X[6] = {-1, -1, 1, 1, 1, -1};
const float Y[6] = {-1, 1, -1, 1, -1, 1};

// Returns the quad billboard coordinate for the current vertex. This quad is placed at (-1,-1) and
// extends to (1,1). Draw mode is CCW front faces. Obviously draw six vertices only.
vec2 vertex_pos()
{
    float x = X[gl_VertexIndex];
    float y = Y[gl_VertexIndex];

    return vec2(x, y);
}

// Returns the quad billboard coordinate for the current vertex. Texture coordinates start at (0,0)
// and evenly texture to (1,1). Obviously draw six vertices only.
vec2 vertex_tex()
{
    float u = U[gl_VertexIndex];
    float v = V[gl_VertexIndex];

    return vec2(u, v);
}
