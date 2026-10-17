#version 330 core

in vec2 uv;
out vec4 fragColor;

uniform sampler2D u_spec;
uniform sampler2D u_colormap;
uniform sampler2D u_colormap2;   // crossfade target
uniform float u_lut_mix;      // 0=u_colormap, 1=u_colormap2
uniform float u_vmin;
uniform float u_vmax;
uniform int u_scale_mode;   // 0=linear, 1=log, 2=mel, 3=bark
uniform float u_f_min;
uniform float u_f_max;
uniform int u_filled_cols;
uniform int u_total_cols;
uniform float u_n_freqs;
uniform float u_t_start;    // view window: start time as fraction [0,1]
uniform float u_t_end;      // view window: end time as fraction [0,1]
uniform float u_fview_min;  // view window: min freq as fraction [0,1]
uniform float u_fview_max;  // view window: max freq as fraction [0,1]
uniform float u_curve_power; // brightness curve exponent (1.0 = linear)
uniform float u_curve_lo;    // lower clamp bound after pow (spectra: 0.15)
uniform float u_curve_span;  // clamp span after pow (spectra: 0.70)
uniform vec3 u_floor;        // canvas background (== LUT flat floor, from styles.BG_CANVAS)

void main() {
    // Map UV through view window
    float u = u_t_start + uv.x * (u_t_end - u_t_start);
    float v = u_fview_min + uv.y * (u_fview_max - u_fview_min);

    // Soft gate: fade to black over ~2 columns at the fill boundary,
    // avoiding the binary pixel-snap flicker of an int-based hard gate.
    float col_f = u * float(u_total_cols);
    float edge_dist = float(u_filled_cols) - col_f;
    if (edge_dist <= 0.0) {
        fragColor = vec4(u_floor, 1.0);
        return;
    }

    // Y-axis frequency warping
    float y;
    if (u_scale_mode == 1) {
        // Log scale
        float f_min_safe = max(u_f_min, 1.0);
        float f_val_log  = log(f_min_safe) + v * (log(u_f_max) - log(f_min_safe));
        float f_norm = (exp(f_val_log) - u_f_min) / (u_f_max - u_f_min);
        y = clamp(f_norm, 0.0, 1.0);
    } else if (u_scale_mode == 2) {
        // Mel scale: v is linear in mel space → convert to Hz → normalise
        float mel_min = 2595.0 * log(1.0 + u_f_min / 700.0) / log(10.0);
        float mel_max = 2595.0 * log(1.0 + u_f_max / 700.0) / log(10.0);
        float mel = mel_min + v * (mel_max - mel_min);
        float hz = 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
        y = clamp((hz - u_f_min) / (u_f_max - u_f_min), 0.0, 1.0);
    } else if (u_scale_mode == 3) {
        // Bark scale: v is linear in bark space → convert to Hz → normalise
        float bark_min = 13.0 * atan(0.00076 * u_f_min) + 3.5 * atan(pow(u_f_min / 7500.0, 2.0));
        float bark_max = 13.0 * atan(0.00076 * u_f_max) + 3.5 * atan(pow(u_f_max / 7500.0, 2.0));
        float bark = bark_min + v * (bark_max - bark_min);
        // Newton-Raphson: bark(f) = 13*atan(0.00076*f) + 3.5*atan((f/7500)^2)
        float f = v * (u_f_max - u_f_min) + u_f_min; // initial guess
        for (int i = 0; i < 4; i++) {
            float b = 13.0 * atan(0.00076 * f) + 3.5 * atan(pow(f / 7500.0, 2.0));
            float db_df = 13.0 * 0.00076 / (1.0 + pow(0.00076 * f, 2.0))
                        + 3.5 * 2.0 * f / (7500.0 * 7500.0) / (1.0 + pow(f / 7500.0, 4.0));
            f = f - (b - bark) / max(db_df, 0.0001);
        }
        y = clamp((f - u_f_min) / (u_f_max - u_f_min), 0.0, 1.0);
    } else {
        // Linear
        y = v;
    }

    // 3-tap vertical box filter — anti-alias when n_freqs >> pixel height
    float dy = 1.0 / u_n_freqs;
    float db = texture(u_spec, vec2(u, y)).r;
    db += texture(u_spec, vec2(u, y - dy)).r;
    db += texture(u_spec, vec2(u, y + dy)).r;
    db /= 3.0;

    float t = clamp((db - u_vmin) / (u_vmax - u_vmin), 0.0, 1.0);

    // Brightness curve — spectra: pow + clamp remap; standard palettes: linear (power=1, lo=0, span=1)
    t = pow(t, u_curve_power);
    t = clamp((t - u_curve_lo) / u_curve_span, 0.0, 1.0);

    vec4 cA = texture(u_colormap, vec2(t, 0.5));
    vec4 cB = texture(u_colormap2, vec2(t, 0.5));
    fragColor = mix(cA, cB, u_lut_mix);

    // Soft fade at the fill boundary — 2-column transition zone (toward canvas floor)
    float alpha = min(edge_dist / 2.0, 1.0);
    fragColor.rgb = mix(u_floor, fragColor.rgb, alpha);
}
