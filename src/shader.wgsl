// Vertex shader

// the matrix is split up into 4 vertexes
struct InstanceInput {
    [[location(5)]] model_matrix_0: vec4<f32>;
    [[location(6)]] model_matrix_1: vec4<f32>;
    [[location(7)]] model_matrix_2: vec4<f32>;
    [[location(8)]] model_matrix_3: vec4<f32>;
};

// represents the contents of a buffer resource occupying 
// a single binding slot in the shader’s resource interface. 
// Any structure used as a uniform must be annotated with [[block]]
[[block]] // 1.
struct CameraUniform {
    view_proj: mat4x4<f32>;
};
// specify this is for the group 1 (camera)
[[group(1), binding(0)]] // 2.
var<uniform> camera: CameraUniform;

struct VertexInput {
    // vertext clip coordiantes
    [[location(0)]] position: vec3<f32>;
    [[location(1)]] tex_coords: vec2<f32>;
};

struct VertexOutput {
    // vertext clip coordiantes
    [[builtin(position)]] clip_position: vec4<f32>;
    [[location(0)]] tex_coords: vec2<f32>;
};

[[stage(vertex)]]
fn vs_main(
    model: VertexInput,
    instance: InstanceInput,
) -> VertexOutput {
    let model_matrix = mat4x4<f32>(
        instance.model_matrix_0,
        instance.model_matrix_1,
        instance.model_matrix_2,
        instance.model_matrix_3,
    );
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    // the vector goes on the right, and the matrices gone on the left in order of importance
    // apply the model_matrix before we apply camera_uniform.view_proj. 
    // the camera_uniform.view_proj changes the coordinate system from world space to camera space. 
    // Our model_matrix is a world space transformation, so we don't want to be in camera space when using it.
    out.clip_position = camera.view_proj * model_matrix * vec4<f32>(model.position, 1.0);
    return out;
}

// Fragment shader


[[group(0), binding(0)]]
var t_diffuse: texture_2d<f32>;
[[group(0), binding(1)]]
var s_diffuse: sampler;

[[stage(fragment)]]
fn fs_main(in: VertexOutput) -> [[location(0)]] vec4<f32> {
    return textureSample(t_diffuse, s_diffuse, in.tex_coords);
}
