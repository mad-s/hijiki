struct Quad {
	vec3 origin;
	vec3 edge1;
	vec3 edge2;
};

bool intersectQuad(Ray ray, Quad quad, inout Intersection its) {
	vec3 n = cross(quad.edge1, quad.edge2);
	vec3 ro = ray.origin - quad.origin;
	vec3 q = cross(ro, ray.direction);
	float d = 1./dot(ray.direction, n);
	float u = d*dot(-q, quad.edge2);
	float v = d*dot( q, quad.edge1);

	if (any(bvec4(u<0., u>1., v<0., v>1.))) {
		return false;
	}
	float t = d*dot(-n, ro);
	if (ray.tMin <= t && t <= ray.tMax) {
		its.t = t;
		its.uv = vec2(u,v);
		return true;
	}
	return false;
}

void populateQuadIntersection(Quad quad, inout Intersection its) {
	vec3 t = normalize(quad.edge1);
	vec3 b = normalize(quad.edge2);
	vec3 n = its.n = cross(t, b);
	its.frame = mat3(t, b, n);
}

void sampleQuad(Quad quad, out ShapeQueryRecord sRec) {
	vec3 n = cross(quad.edge1, quad.edge2);
	float area = length(n);
	n /= area;
	sRec.n = n;

	float u = randUniformFloat();
	float v = randUniformFloat();
	sRec.p = quad.origin + u * quad.edge1 + v * quad.edge2;

	sRec.pdf = 1. / area;
}
