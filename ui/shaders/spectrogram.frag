#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_spec;
uniform sampler2D u_colormap;
uniform float u_vmin;
uniform float u_vmax;
uniform int u_log_scale;
uniform float u_f_min;
uniform float u_f_max;
uniform int u_filled_cols;
uniform int u_total_cols;
uniform float u_n_freqs;
uniform float u_t_start;    // view window: start time as fraction [0,1]
uniform float u_t_end;      // view window: end time as fraction [0,1]
uniform float u_fview_min;  // view window: min freq as fraction [0,1]
uniform float u_fview_max;  // view window: max freq as fraction [0,1]

void main() {
    // Map UV through view window
    float u = u_t_start + uv.x * (u_t_end - u_t_start);
    float v = u_fview_min + uv.y * (u_fview_max - u_fview_min);

    // Soft gate: fade to black over ~2 columns at the fill boundary,
    // avoiding the binary pixel-snap flicker of an int-based hard gate.
    float col_f = u * float(u_total_cols);
    float edge_dist = float(u_filled_cols) - col_f;
    if (edge_dist <= 0.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // Y-axis: log or linear
    float y;
    if (u_log_scale == 1) {
        float f_min_safe = max(u_f_min, 1.0);
        float f_val_log  = log(f_min_safe) + v * (log(u_f_max) - log(f_min_safe));
        float f_norm = (exp(f_val_log) - u_f_min) / (u_f_max - u_f_min);
        y = clamp(f_norm, 0.0, 1.0);
    } else {
        y = v;
    }

    // 3-tap vertical box filter — anti-alias when n_freqs >> pixel height
    float dy = 1.0 / u_n_freqs;
    float db = texture(u_spec, vec2(u, y)).r;
    db += texture(u_spec, vec2(u, y - dy)).r;
    db += texture(u_spec, vec2(u, y + dy)).r;
    db /= 3.0;

    float t = clamp((db - u_vmin) / (u_vmax - u_vmin), 0.0, 1.0);
    t = pow(t, 0.5);
    t = clamp((t - 0.15) / 0.7, 0.0, 1.0);

    fragColor = texture(u_colormap, vec2(t, 0.5));

    // Soft fade at the fill boundary — 2-column transition zone
    float alpha = min(edge_dist / 2.0, 1.0);
    fragColor.rgb *= alpha;
}
