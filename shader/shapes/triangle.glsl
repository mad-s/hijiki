struct Vertex {
	vec4 pos_u;
	vec4 norm_v;
};

layout(set = 0, binding = BINDING_TRIANGLES) buffer Triangles {
	uint triangles[];
};

layout(set = 0, binding = BINDING_VERTICES) buffer Vertices {
	Vertex vertices[];
};


bool intersectTriangle(Ray ray, uint ix, inout Intersection its) {
	Vertex a = vertices[triangles[3*ix+0]];
	Vertex b = vertices[triangles[3*ix+1]];
	Vertex c = vertices[triangles[3*ix+2]];
	vec3 ab = b.pos_u.xyz - a.pos_u.xyz;
	vec3 ac = c.pos_u.xyz - a.pos_u.xyz;

	vec3 n = cross(ab, ac);
	vec3 ro = ray.origin - a.pos_u.xyz;
	vec3 q = cross(ro, ray.direction);
	float d = 1./dot(ray.direction, n);
	float u = d*dot(-q, ac);
	float v = d*dot( q, ab);

	if (u<0. || v<0. || u+v > 1.) {
		return false;
	}

	float t = d*dot(-n, ro);
	if (ray.tMin <= t && t <= ray.tMax) {
		its.t = t;
		its.uv = vec2(u,v);
		its.n = normalize(n);
		/*
		vec3 lambda = vec3(1.-u-v,u,v);
		its.n = normalize(
			a.norm_v.xyz * lambda[0] +
			b.norm_v.xyz * lambda[1] +
			c.norm_v.xyz * lambda[2]);
		its.uv = normalize(
			a.norm_v.xyz * lambda[0] +
			b.norm_v.xyz * lambda[1] +
			c.norm_v.xyz * lambda[2]);
		//*/
		return true;
	}
	return false;
}

void populateTriangleIntersection(uint ix, inout Intersection its) {
	vec3 lambda = vec3(1.-its.uv.x-its.uv.y,its.uv);
	Vertex a = vertices[triangles[3*ix+0]];
	Vertex b = vertices[triangles[3*ix+1]];
	Vertex c = vertices[triangles[3*ix+2]];

	its.n = normalize(
		a.norm_v.xyz * lambda[0] +
		b.norm_v.xyz * lambda[1] +
		c.norm_v.xyz * lambda[2]);
	its.uv = 
		vec2(a.pos_u.w, a.norm_v.w) * lambda[0] +
		vec2(b.pos_u.w, b.norm_v.w) * lambda[1] +
		vec2(c.pos_u.w, c.norm_v.w) * lambda[2];
	// TODO: proper tangent space
	vec3 t,bt;
	if (abs(its.n.x) > abs(its.n.y)) {
		bt = vec3(0.,1.,0.);
	} else {
		bt = vec3(1.,0.,0.);
	}
	t = normalize(cross(its.n, bt));
	bt = cross(its.n, t);
	its.frame = mat3(t, bt, its.n);
}


void sampleTriangle(uint ix, out ShapeQueryRecord sRec) {
	Vertex a = vertices[triangles[3*ix+0]];
	Vertex b = vertices[triangles[3*ix+1]];
	Vertex c = vertices[triangles[3*ix+2]];
	vec3 ab = b.pos_u.xyz - a.pos_u.xyz;
	vec3 ac = c.pos_u.xyz - a.pos_u.xyz;

	vec3 n = cross(ab, ac);
	float area = length(n)/2.;
	n /= 2.*area;

	vec3 lambda = randBarycentric();
	sRec.n = normalize(
		a.norm_v.xyz * lambda[0] +
		b.norm_v.xyz * lambda[1] +
		c.norm_v.xyz * lambda[2]);
	sRec.p =
		a.pos_u.xyz * lambda[0] +
		b.pos_u.xyz * lambda[1] +
		c.pos_u.xyz * lambda[2];
	sRec.pdf = 1.0 / area;
}


