layout(set = 0, binding = 8) buffer DiffuseMaterials {
	 DiffuseMaterial diffuseMaterials[];
};

layout(set = 0, binding = 9) buffer MirrorMaterials {
	 MirrorMaterial mirrorMaterials[];
};

layout(set = 0, binding = 10) buffer DielectricMaterials {
	 DielectricMaterial dielectricMaterials[];
};

layout(set = 0, binding = 11) buffer EmissiveMaterials {
	 EmissiveMaterial emissiveMaterials[];
};

layout(set = 0, binding = 12) buffer PortalMaterials {
	 PortalMaterial portalMaterials[];
};
