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
	float t = d*dot(-n, ro);
	if (u < 0. || u > 1. || v < 0. || v > 1.) {
		return false;
	}
	if (ray.tMin <= t && t <= ray.tMax) {
		its.t = t;
		return true;
	}
	return false;
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
