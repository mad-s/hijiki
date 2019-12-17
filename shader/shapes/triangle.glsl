struct Vertex {
	vec4 pos_u;
	vec4 norm_v;
};

bool intersectTriangle(Ray ray, Vertex a, Vertex b, Vertex c, inout Intersection its) {
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
		vec3 lambda = vec3(1.-u-v,u,v);
		its.n = normalize(
			a.norm_v.xyz * lambda[0] +
			b.norm_v.xyz * lambda[1] +
			c.norm_v.xyz * lambda[2]);
		//its.n = normalize(n);
		return true;
	}
	return false;
}


void sampleTriangle(Vertex a, Vertex b, Vertex c, out ShapeQueryRecord sRec) {
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


