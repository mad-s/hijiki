vec4 quaternionMult(vec4 qa, vec4 qb) {
	vec4 result;
	result.w = qa.w*qb.w - dot(qa.xyz, qb.xyz);
	result.xyz = cross(qa.xyz, qb.xyz) + qa.xyz * qb.w + qb.xyz * qa.w;
	return result;
}

vec4 quaternionFromTo(vec3 a, vec3 b) {
	vec4 result;
	result.xyz = cross(a, b);
	result.w = 1+dot(a, b);
	return normalize(result);
}

vec3 quaternionRotate(vec3 v, vec4 r) {
	vec4 tmp = quaternionMult(r, vec4(v, 0.));
	r.xyz = -r.xyz;
	return quaternionMult(tmp, r).xyz;
}
