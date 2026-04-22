
class DatasetConstants():
    def __init__(self, base_path, dataset_name):

        self.base_path = base_path
        self.dataset_name = dataset_name
        self.data_path = f"{base_path}/{dataset_name}"

        self.DATA_PATH = {
            "Brain": f"{base_path}/MedAD/Brain_AD",
            "Liver": f"{base_path}/MedAD/Liver_AD",
            "Retina": f"{base_path}/MedAD/Retina_RESC_AD",
            "Colon_clinicDB": f"{base_path}/Colon/CVC-ClinicDB",
            "Colon_colonDB": f"{base_path}/Colon/CVC-ColonDB",
            "Colon_cvc300": f"{base_path}/Colon/CVC-300",
            "Colon_Kvasir": f"{base_path}/Colon/Kvasir",
            "BTAD": f"{base_path}/BTech_Dataset_transformed",
            "MPDD": f"{base_path}/MPDD",
            "MVTec": f"{base_path}/mvtec_ad",
            "VisA": f"{base_path}/VisA",
        }

        self.CLASS_NAMES = {
            "Brain": ["Brain"],
            "Liver": ["Liver"],
            "Retina": ["Retina"],
            "Colon_clinicDB": ["Colon_clinicDB"],
            "Colon_colonDB": ["Colon_colonDB"],
            "Colon_Kvasir": ["Kvasir"],
            "Colon_cvc300": ["CVC-300"],
            "MVTec": [
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
            "VisA": [
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
            "MPDD": [
                "connector",
                "tubes",
                "metal_plate",
                "bracket_white",
                "bracket_brown",
                "bracket_black",
            ],
            "BTAD": ["01", "02", "03"],
        }

        self.DOMAINS = {
            "VisA": "Industrial",
            "BTAD": "Industrial",
            "MPDD": "Industrial",
            "MVTec": "Industrial",
            "Brain": "Medical",
            "Liver": "Medical",
            "Retina": "Medical",
            "Colon_clinicDB": "Medical",
            "Colon_colonDB": "Medical",
            "Colon_Kvasir": "Medical",
            "Colon_cvc300": "Medical",
        }
        
        self.REAL_NAMES = {
            "Brain": {"Brain": "scan"},
            "Liver": {"Liver": "scan"},
            "Retina": {"Retina": "scan"},
            "MVTec": {
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
                "transistor": "a three-legged transistor placed vertically",
                "toothbrush": "toothbrush head",
                "wood": "wood surface",
                "zipper": "a black zipper",
            },
            "VisA": {
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
            "Colon_clinicDB": {
                "Colon_clinicDB": "colon endoscopy image",
            },
            "Colon_colonDB": {
                "Colon_colonDB": "colon endoscopy image",
            },
            "Colon_cvc300": {"CVC-300": "colon endoscopy image"},
            "Colon_Kvasir": {"Kvasir": "colon endoscopy image"},
            "MPDD": {
                "connector": "metal clamps with black adjustment knobs",
                "tubes": "scattered metal objects",
                "metal_plate": "blue rectangular metal plate with a notch on one side",
                "bracket_white": "white, elongated triangular metal bracket with a smooth, matte finish",
                "bracket_brown": "brown L-shaped metal bracket with smooth, glossy finish and multiple mounting holes along its arms",
                "bracket_black": "black ornamental metal bracket with spiral design attached to a rectangular frame",
            },
            "BTAD": {
                "01": "Bright concentric rings in neon yellow and blue tones against a dark blue background, resembling a stylized wave or energy field radiating outward.",
                "02": "vertical fabric lines in warm, dusty pink and beige tones",
                "03": "oval concentric circular rings in gradient shades of blue and white",
            },
        }
        
        self.PROMPTS = {
            "prompt_normal": ["{}", "a {}", "the {}"],
            "prompt_abnormal": [
                "a damaged {}",
                "a broken {}",
                "a {} with flaw",
                "a {} with defect",
                "a {} with damage",
            ],
            "prompt_templates": [
                "{}.",
                "a photo of {}.",
            ],
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
    dataset_name = "MVTec"
    constants = DatasetConstants(base_path, dataset_name)
    print(constants.get_data_path())
    print(constants.get_class_names())
    print(constants.get_domain())
    print(constants.get_real_names())
    print(constants.get_real_name("bottle"))
    print(constants.get_prompts())