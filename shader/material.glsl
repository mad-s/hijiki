layout(set = 0, binding = 7) buffer DiffuseMaterials {
	 DiffuseMaterial diffuseMaterials[];
};

layout(set = 0, binding = 8) buffer MirrorMaterials {
	 MirrorMaterial mirrorMaterials[];
};

layout(set = 0, binding = 9) buffer DielectricMaterials {
	 DielectricMaterial dielectricMaterials[];
};

layout(set = 0, binding = 10) buffer EmissiveMaterials {
	 EmissiveMaterial emissiveMaterials[];
};
