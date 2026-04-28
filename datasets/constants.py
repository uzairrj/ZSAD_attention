class DatasetConstants:
    def __init__(self, base_path, dataset_name):
        self.base_path = base_path
        self.dataset_name = dataset_name.lower()

        self.DATA_PATH = {
            "brain": f"{base_path}/MedAD/Brain_AD",
            "liver": f"{base_path}/MedAD/Liver_AD",
            "retina": f"{base_path}/MedAD/Retina_RESC_AD",
            "colon_clinicdb": f"{base_path}/Colon/CVC-ClinicDB",
            "colon_colondb": f"{base_path}/Colon/CVC-ColonDB",
            "colon_cvc300": f"{base_path}/Colon/CVC-300",
            "colon_kvasir": f"{base_path}/Colon/Kvasir",
            "btad": f"{base_path}/btad",
            "mpdd": f"{base_path}/MPDD",
            "mvtec": f"{base_path}/mvtec",
            "visa": f"{base_path}/visa",
        }

        self.CLASS_NAMES = {
            "brain": ["Brain"],
            "liver": ["Liver"],
            "retina": ["Retina"],
            "colon_clinicdb": ["Colon_clinicDB"],
            "colon_colondb": ["Colon_colonDB"],
            "colon_kvasir": ["Kvasir"],
            "colon_cvc300": ["CVC-300"],
            "mvtec": [
                "bottle",
                "cable",
                "capsule",
                "carpet",
                "grid",
                "hazelnut",
                "leather",
                "metal_nut",
                "pill",
                "screw",
                "tile",
                "transistor",
                "toothbrush",
                "wood",
                "zipper",
            ],
            "visa": [
                "candle",
                "pcb3",
                "capsules",
                "pipe_fryum",
                "pcb4",
                "macaroni2",
                "pcb2",
                "chewinggum",
                "macaroni1",
                "cashew",
                "fryum",
                "pcb1",
            ],
            "mpdd": [
                "connector",
                "tubes",
                "metal_plate",
                "bracket_white",
                "bracket_brown",
                "bracket_black",
            ],
            "btad": ["01", "02", "03"],
        }

        self.DOMAINS = {
            "visa": "Industrial",
            "btad": "Industrial",
            "mpdd": "Industrial",
            "mvtec": "Industrial",
            "brain": "Medical",
            "liver": "Medical",
            "retina": "Medical",
            "colon_clinicdb": "Medical",
            "colon_colondb": "Medical",
            "colon_kvasir": "Medical",
            "colon_cvc300": "Medical",
        }

        self.REAL_NAMES = {
            "brain": {"Brain": "scan"},
            "liver": {"Liver": "scan"},
            "retina": {"Retina": "scan"},
            "mvtec": {
                "bottle": "dark bottle",
                "cable": "top view of three cables",
                "capsule": "black and orange capsule",
                "carpet": "gray carpet",
                "grid": "metal or plastic mesh",
                "hazelnut": "single brown hazelnut",
                "leather": "brown leather",
                "metal_nut": "metal nut which has four notched edges",
                "pill": "oval white pill with small red speckles and the letters 'FF' engraved",
                "screw": "screw",
                "tile": "speckled tile surface",
                "transistor": "three-legged transistor placed vertically",
                "toothbrush": "toothbrush head",
                "wood": "wood surface",
                "zipper": "black zipper",
            },
            "visa": {
                "candle": "candle",
                "pcb3": "infrared sensor pcb module",
                "capsules": "capsules",
                "pipe_fryum": "pipe-shaped fryum",
                "pcb4": "battery charging pcb module",
                "macaroni2": "scattered yellow macaroni",
                "pcb2": "integrated circuits board",
                "chewinggum": "chewing gum",
                "macaroni1": "orange macaroni",
                "cashew": "cashew nut",
                "fryum": "wheel-shaped fryum snack",
                "pcb1": "dual ultrasonic distance sensor pcb module",
            },
            "colon_clinicdb": {
                "Colon_clinicDB": "colon endoscopy image",
            },
            "colon_colondb": {
                "Colon_colonDB": "colon endoscopy image",
            },
            "colon_cvc300": {"CVC-300": "colon endoscopy image"},
            "colon_kvasir": {"Kvasir": "colon endoscopy image"},
            # "mpdd": {
            #     "connector": "metal clamps with black adjustment knobs",
            #     "tubes": "scattered metal objects",
            #     "metal_plate": "blue rectangular metal plate with a notch on one side",
            #     "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
            #     "bracket_brown": "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting holes along its arms",
            #     "bracket_black": "black ornamental metal bracket with spiral design attached to a rectangular frame",
            # },
            "mpdd": {
                "connector": "metal connector part with black knobs",
                "tubes": "metal tubes",
                "metal_plate": "blue rectangular metal plate with a side notch",
                "bracket_white": "white painted metal bracket",
                "bracket_brown": "brown painted metal bracket with mounting holes",
                "bracket_black": "black painted metal bracket with curved ornamental shape",
            },
            "btad": {
                "01": "Bright concentric rings in neon yellow and blue tones against a dark blue background, resembling a stylized wave or energy field radiating outward.",
                "02": "vertical fabric lines in warm, dusty pink and beige tones",
                "03": "oval concentric circular rings in gradient shades of blue and white",
            },
        }

        self.PROMPTS = {
            "prompt_normal": ['{}', 'flawless {}', 'perfect {}', 'unblemished {}', 
                              '{} without flaw', '{} without defect', '{} without damage'],
            "prompt_abnormal": ['damaged {}', 'broken {}', '{} with flaw', 
                                '{} with defect', '{} with damage'],
            "prompt_templates": ['a bad photo of a {}.', 
                                 'a low resolution photo of the {}.', 
                                 'a bad photo of the {}.', 
                                 'a cropped photo of the {}.', 
                                 'a bright photo of a {}.', 
                                 'a dark photo of the {}.', 
                                 'a photo of my {}.', 
                                 'a photo of the cool {}.', 
                                 'a close-up photo of a {}.', 
                                 'a black and white photo of the {}.', 
                                 'a bright photo of the {}.', 
                                 'a cropped photo of a {}.', 
                                 'a jpeg corrupted photo of a {}.', 
                                 'a blurry photo of the {}.', 
                                 'a photo of the {}.', 
                                 'a good photo of the {}.', 
                                 'a photo of one {}.', 
                                 'a close-up photo of the {}.', 
                                 'a photo of a {}.', 
                                 'a low resolution photo of a {}.', 
                                 'a photo of a large {}.', 
                                 'a blurry photo of a {}.', 
                                 'a jpeg corrupted photo of the {}.', 
                                 'a good photo of a {}.', 
                                 'a photo of the small {}.', 
                                 'a photo of the large {}.', 
                                 'a black and white photo of a {}.', 
                                 'a dark photo of a {}.', 
                                 'a photo of a cool {}.',
                                'a photo of a small {}.', 
                                'there is a {} in the scene.', 
                                'there is the {} in the scene.', 
                                'this is a {} in the scene.', 
                                'this is the {} in the scene.', 
                                'this is one {} in the scene.']
        }

    def get_data_path(self):
        return self.DATA_PATH[self.dataset_name]

    def get_class_names(self):
        return self.CLASS_NAMES[self.dataset_name]

    def get_domain(self):
        return self.DOMAINS[self.dataset_name]

    def get_real_names(self):
        return self.REAL_NAMES[self.dataset_name]

    def get_real_name(self, class_name):
        return self.REAL_NAMES[self.dataset_name][class_name]

    def get_prompts(self):
        return self.PROMPTS


if __name__ == "__main__":
    base_path = "/path/to/datasets"
    dataset_name = "MVTec"   # works, gets converted to "mvtec"
    constants = DatasetConstants(base_path, dataset_name)

    print(constants.get_data_path())
    print(constants.get_class_names())
    print(constants.get_domain())
    print(constants.get_real_names())
    print(constants.get_real_name("bottle"))
    print(constants.get_prompts())