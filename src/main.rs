extern crate wgpu;

extern crate shaderc;

use std::any::TypeId;
use std::collections::HashMap;

use std::fmt::Write;

use bvh::nalgebra as na;

type Vec2 = na::Vector2<f32>;
type Vec3 = na::Vector3<f32>;
type Vec4 = na::Vector4<f32>;


unsafe trait ToGpu {
    fn gpu_type_name() -> String;
    fn struct_def() -> String {
        "".into()
    }
}
macro_rules! gpu_primitive {
    ($t:ty = $g:ident) => {
        unsafe impl ToGpu for $t {
            fn gpu_type_name() -> String {
                stringify!($g).into()
            }
        }
    }
}
gpu_primitive!(f32 = float);
gpu_primitive!(Vec2 = vec2);
//gpu_primitive!(Vec3 = vec3);
gpu_primitive!(Vec4 = vec4);

macro_rules! gpu_struct {
    (struct $T:ident {
        $(
            $name:ident : $t:ty,
        )*
    }) => {
        struct $T {
            $($name:$t,)*
        }
        unsafe impl ToGpu for $T {
            fn gpu_type_name() -> String {
                stringify!($T).into()
            }
            fn struct_def() -> String {
                let mut res : String = "".into();
                res.push_str(concat!("struct ", stringify!($T), " {\n"));
                $(
                    res.push_str("    ");
                    res.push_str(&<$t as ToGpu>::gpu_type_name());
                    res.push_str(concat!(" ", stringify!($name), ";\n"));
                )*
                res.push_str("}\n");
                res
            }
        }
    }
}


struct RegisteredTraitImpl {
    name: String,
    tag: usize,
    use_index: bool,
}


macro_rules! stringify_last {
    /*
    (; $x:ident $($xs:ident)*) => {
        stringify!($x)
    };
    ($x:ident $($tail:ident)* ; $($xs:ident)*) => {
        stringify_last!($($tail)* ; $x $($xs)*)
    };
    ($($xs:ident)*) => {
        stringify_last!($($xs)* ;)
    };
    */
    ($x:ident) => {
        stringify!($x)
    };
    ($x:ident $($xs:ident)+) => {
        stringify_last!($($xs)+)
    };
}
macro_rules! gpu_trait {
    (trait $Trait:ident registry $Registry:ident {
        $($fres:ident $fname:ident (self $(, $($argx:ident)*)* );)*
    }) => {gpu_trait!(
        trait $Trait registry $Registry {
        $($fres $fname (self $(, $($argx)*)* );)*
        ;
        }
    );};
    (trait $Trait:ident registry $Registry:ident {
        $($fres:ident $fname:ident (self $(, $($argx:ident)*)* );)*
        ;
        $($extra_trait_item:tt)*
    }
    ) => {
        trait $Trait : ToGpu {
            $(
                fn $fname() -> String {
                    "// TODO".into()
                }
            )*
            fn custom_binding() -> Option<String> {
                None
            }
            $($extra_trait_item)*
        }

        struct $Registry {
            code: String,
            registered_impls: HashMap<TypeId, RegisteredTraitImpl>,
            last_tag: usize,
        }
        impl $Registry {
            const SIGNATURES : &'static [(&'static str, &'static str, &'static [&'static [&'static str]])] = {
                &[
                    $(
                    (stringify!($fres),stringify!($fname),&[$(&[$(stringify!($argx),)*],)*]),
                    )*
                ]
            };
            fn new() -> Self {
                $Registry {
                    code: String::new(),
                    registered_impls: HashMap::new(),
                    last_tag: 1,
                }
            }
            fn register<T: $Trait + 'static>(&mut self) {
                let name = <T as ToGpu>::gpu_type_name();
                let typeid = TypeId::of::<T>();
                assert!(!self.registered_impls.contains_key(&typeid));

                let (binding_code, self_type, use_index) : (String, &str, bool) = match T::custom_binding() {
                    Some(code) => (code, "uint", true),
                    None => (format!(
"layout(set = 0, binding = {name}_binding) buffer {name}_buffer {{
    {name} {name}_array[];
}}\n", name=name), &name, false),
                };
                self.registered_impls.insert(typeid, RegisteredTraitImpl {
                    name: name.to_owned(),
                    use_index,
                    tag: self.last_tag,
                });

                writeln!(self.code, "#define {}_tag {}", name, self.last_tag);
                self.last_tag += 1;
                self.code.push_str(&T::struct_def());
                self.code.push_str(&binding_code);
                $(
                    writeln!(self.code, 
concat!(stringify!($fres), " {name}_", stringify!($fname), "({self_type} self" $(,"," $(," ",stringify!($argx))*)*, ") {{
{code}
}}"),
name=&name, self_type=&self_type, code=T::$fname()
);
                )*
            }
            fn finalize(mut self) -> String {
                writeln!(self.code, concat!("#define ", stringify!($Trait), "_ref uint"));
                $(
                    $(
                        dbg!(stringify_last!($($argx)*));
                    )*
                    writeln!(self.code, concat!(stringify!($fres), " ", stringify!($Trait), "_", stringify!($fname), "(", stringify!($Trait), "_ref self" $(,"," $(," ",stringify!($argx))*)*, ") {{
    uint tag = self >> 24;
    uint index = self & ((1 << 24) - 1);
    switch (tag) {{"));
                    for i in self.registered_impls.values() {
                        let name = &i.name;
                        let self_arg = if i.use_index {format!("index")} else {format!("{name}_array[index]", name=name)};
                        writeln!(self.code,       "       case {name}_tag:", name=i.name);
                        writeln!(self.code, concat!("           return {name}_", stringify!($fname), "({self_arg}" $(, ", ", stringify_last!($($argx)*))*, ");"), name=i.name, self_arg=self_arg);

                    }
                    writeln!(self.code, "    }}\n}}");
                )*
                self.code
            }
        }
    }
}

gpu_trait! {
    trait Material registry MaterialRegistry {
        vec3 sampleBSDF(self, vec3 wi, Intersection its, out vec3 wo);
        vec3 evalBSDF(self, vec3 wi, Intersection its, vec3 wo);
    }
}

gpu_struct! {
    struct DiffuseMaterial {
        color_pad: Vec4,
    }
}


impl Material for DiffuseMaterial {
    fn sampleBSDF() -> String {
        "// sampleBSDF() code here".into()
    }

    fn evalBSDF() -> String {
        "// evalBSDF() code here".into()
    }
}


gpu_trait! {
    trait Shape registry ShapeRegistry {
        vec3 sample(self);
        void intersect(self, vec3 ro, vec3 rd, inout Intersection its);
    }
}

gpu_struct! {
    struct Sphere {
        center_radius: Vec4,
    }
}

struct Triangle {
    indices: [u32; 3],
}
unsafe impl ToGpu for Triangle {
    fn gpu_type_name() -> String {
        "Triangle".into()
    }
}

impl Shape for Triangle {
    fn sample() -> String {
        "// TODO".into()
    }

    fn custom_binding() -> Option<String> {
        Some(
"layout(set = 0, binding = Triangle_binding) buffer Triangle_buffer {
    uint Triangle_array[];
}\n".into())
    }
}

impl Shape for Sphere {
    fn sample() -> String {
        "// TODO".into()
    }
}


fn main() {
    /*
    let mut material_reg = MaterialRegistry::new();
    material_reg.register::<DiffuseMaterial>();
    dbg!(MaterialRegistry::SIGNATURES);
    println!("{}", material_reg.finalize());
    */

    let mut shape_reg = ShapeRegistry::new();
    shape_reg.register::<Sphere>();
    shape_reg.register::<Triangle>();
    println!("{}", shape_reg.finalize());
}

