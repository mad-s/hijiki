struct DiffuseCheckerboardMaterial {
        vec4 color_a_scale_u;
        vec4 color_b_scale_v;
};

vec3 getCheckerboardTexture(DiffuseCheckerboardMaterial mat, vec2 uv) {
        uv = fract(0.5 * uv / vec2(mat.color_a_scale_u.w, mat.color_b_scale_v.w));
        if (uv.x < 0.5 ^^ uv.y < 0.5) {
                return mat.color_b_scale_v.rgb;
        } else {
                return mat.color_a_scale_u.rgb;
        }
}
