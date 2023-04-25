# Dependencies
import json 
import sys 
import time 
from pathlib import Path 
from typing import Sequence, Tuple 
import cv2 
import matplotlib.pyplot as plt 
import numpy as np 
import torch 
from torchmetrics import detection as torchmetrics_detection 
from openvino.runtime import Core 
from openvino.tools.pot import Metric, DataLoader, IEEngine, load_model, save_model, compress_model_weights, create_pipeline
sys.path.append('.../utils')

# Get the model
ir_path = Path('intel/person-detection-retail-0013/FP32/person-detection-retail-0013.xml')

# Load the model 
ie = Core()
model = ie.read_model(model = ir_path)
compiled_model = ie.compile_model(model=model, device_name='CPU')
input_layer = compiled_model.input(0)
output_layer = compiled_model.output(0)
input_size = input_layer.shape
_, _, input_height, input_width = input_size

# Configration
class DetectionDataLoader(DataLoader):
    def __init__(self, basedir: str, target_size: Tuple[int, int]):
        self.images = sorted(Path(basedir).glob("*.jpg"))
        self.target_size = target_size
        with open(f'{basedir}/annotation_person_train.json') as f:
            self.annotations = json.load(f)
        self.image_ids = {
            Path(item['file_name']).name: item['id']
            for item in self.annotations['images']
        }

        for image_filename in self.images:
            annotations = [
                item for item in self.annotations['annotations']
                if item['image_id'] == self.image_ids[Path(image_filename).name]
            ]
            assert(len(annotations) !=0), f'No annotations found for image id {image_filename}'

        print(f'Created dataset with {len(self.images)} items. Data dir: {basedir}')

    def __getitem__(self, index):
        image_path = self.images[index]
        image = cv2.imread(str(image_path))
        image = cv2.resize(image, self.target_size)
        image_id = self.image_ids[Path(image_path).name]
        image_info = [image for image in self.annotations['images'] if image['id'] == image_id][0]
        image_annotations = [item for item in self.annotations['annotations'] if item['image_id'] == image_id]
    
        target_annotations =[]
        for annotation in image_annotations:
            xmin, ymin, width, height = annotation['bbox']
            xmax = xmin + width
            ymax = ymin + height
            xmin /= image_info['width']
            ymin /= image_info['height']
            xmax /= image_info['width']
            ymax /= image_info['height']
            target_annotation = {}
            target_annotation['category_id'] = annotation['category_id']
            target_annotation['image_width'] = image_info['width']
            target_annotation['image_height'] = image_info['height']
            target_annotation['bbox'] = [xmin, ymin, xmax, ymax]
            target_annotations.append(target_annotation)

        item_annotation = (index, target_annotations)
        input_image = np.expand_dims(image.transpose(2, 0, 1), axis = 0).astype(np.float32)
        
        return (item_annotation, input_image, {'filename': str(image_path), 'shape': image.shape})
    
    def __len__(self):
        return len(self.images) 

# Metric
class MAPMetric(Metric):
    def __init__(self, map_value ='map'):
        assert(
            map_value in torchmetrics_detection.mean_ap.MARMetricResults.__slots__
            + torchmetrics_detection.mean_ap.MAPMetricResults.__slots__
        )

        self._name = map_value
        self.metric = torchmetrics_detection.mean_ap.MeanAveragePrecision()
        super().__init__()

    @property
    def value(self):
        return {self._name: [0]}
    
    @property
    def avg_value(self):
        return {self._name: self.metric.compute()[self._name].item()}
    
    def update(self, output, target):
        targetboxes = []
        targetlabels = []
        predboxes = []
        predlabels = []
        scores = []

        image_width = target[0][0]['image_width']
        image_height = target[0][0]['image_height']

        for single_target in target[0]:
            txmin, tymin, txmax, tymax = single_target['bbox']
            category = single_target['category_id']
            txmin *= image_width
            txmax *= image_width
            tymin *= image_height
            tymax *= image_height

            targetbox = [round(txmin), round(tymin), round(txmax), round(tymax)]
            targetboxes.append(targetbox)
            targetlabels.append(category)

        for single_output in output:
            for pred in single_output[0, 0, ::]:
                image_id, label, conf, xmin, ymin, xmax, ymax = pred
                xmin *= image_width
                xmax *= image_width
                ymin *= image_height
                ymax *= image_height

                predbox = [round(xmin), round(ymin), round(xmax), round(ymax)]
                predboxes.append(predbox)
                predlabels.append(label)
                scores.append(conf)

        preds = [dict(
            boxes = torch.Tensor(predboxes).float(),
            labels = torch.Tensor(predlabels).short(),
            scores = torch.Tensor(scores),
        )]

        targets = [dict(
            boxes = torch.Tensor(targetboxes).float(),
            labels = torch.Tensor(targetlabels).float(),
        )]

        self.metric.update(preds, targets)

    def reset(self):
        self.metric.reset()

    def get_attributes(self):
        return {self._name: {'direction': 'higher-better', 'type': 'mAP'}}
    

# Quantization Config
model_config = {
    'model_name': ir_path.stem,
    'model': ir_path,
    'weights': ir_path.with_suffix('.bin')
}

engine_config = {'device': 'CPU'}

default_algorithms = [
    {
    'name': 'DefaultQuantization',
    'stat_subset_size': 300,
    'params': {'target_device': 'ANY', 'preset': 'mixed'},
    }
]

print(f'model_config: {model_config}')


# Run Quantization Pipeline
# Create the data loader
data_loader = DetectionDataLoader(basedir='data/person_detection', target_size = (input_width, input_height))

# Load the model
ir_model = load_model(model_config=model_config)

# Initialize the metric
metric = MAPMetric(map_value = 'map')

# Initialize the engine for metric calculation and statistics collection
engine = IEEngine(config = engine_config, data_loader = data_loader, metric = metric)

# Create a pipeline of compression algorithms
pipeline = create_pipeline(default_algorithms, engine)

# Execute the pipeline to quantize the model 
algorithm_name = pipeline.algo_seq[0].name

print(f"Executing POT pipeline on {model_config['model']} with {algorithm_name}")
start_time = time.perf_counter()
compressed_model = pipeline.run(ir_model)
end_time = time.perf_counter()
print(f'Quantization finished in {end_time - start_time:.2f} seconds')

# Compress model weights to quantized precision
compress_model_weights(compressed_model)

# Save the compressed model to the desired path
preset = pipeline._algo_seq[0].config['preset']
algorithm_name = pipeline._algo_seq[0].name
compressed_model_paths = save_model(model = compressed_model,
                                  save_path = 'optimized_model',
                                  model_name = f'{ir_model.name}_{preset}_{algorithm_name}',)

compressed_model_path = compressed_model_paths[0]['model']
print('The quantized model is stored at', compressed_model_path)

# Compare Metric of Floating Poin and Quantized Model
ir_model = load_model(model_config = model_config)
evaluation_pipeline = create_pipeline(algo_config = dict(), engine = engine)
print('Evaluating original IR model...')
original_metric = evaluation_pipeline.evaluate(ir_model)

print('Evaluating quantized IR model...')
quantized_metric = pipeline.evaluate(compressed_model)

if original_metric:
    for key, value in original_metric.items():
        print(f"The {key} score of the original FP16 model is {value:.5f}")

if quantized_metric:
    for key, value in quantized_metric.items():
        print(f"The {key} score of the quantized INT8 model is {value:.5f}")

def draw_boxes_on_image(box: Sequence[float], image: np.ndarray, color: str, scale: bool = True):
    colors = {"red": (255, 0, 64), "green": (0, 255, 0), "yellow": (255, 255, 128)}
    assert color in colors, f"{color} is not defined yet. Defined colors are: {colors}"
    image_height, image_width, _ = image.shape
    x_min, y_min, x_max, y_max = box
    if scale:
        x_min *= image_width
        x_max *= image_width
        y_min *= image_height
        y_max *= image_height

    image = cv2.rectangle(img=image, pt1=(round(x_min), round(y_min)), pt2=(round(x_max), round(y_max)), color=colors[color], thickness=2,)
    return image

map_value = "map"
confidence_threshold = 0.5
num_images = 4

# FP prediction
fp_model = ie.read_model(model=ir_path)
fp_compiled_model = ie.compile_model(model=fp_model, device_name="CPU")
input_layer_fp = fp_compiled_model.input(0)
output_layer_fp = fp_compiled_model.output(0)

# INT8 prediction
int8_model = ie.read_model(model=compressed_model_path)
int8_compiled_model = ie.compile_model(model=int8_model, device_name="CPU")
input_layer_int8 = int8_compiled_model.input(0)
output_layer_int8 = int8_compiled_model.output(0)

fig, axs = plt.subplots(nrows=num_images, ncols=3, figsize=(16, 14), squeeze=False)
for i in range(num_images):
    annotation, input_image, metadata = data_loader[i]
    image = cv2.cvtColor(
        src=cv2.imread(filename=metadata["filename"]), code=cv2.COLOR_BGR2RGB
    )
    orig_image = image.copy()
    resized_image = cv2.resize(image, (input_width, input_height))
    target_annotation = annotation[1]

    fp_res = fp_compiled_model([input_image])[output_layer_fp]
    
    fp_metric = MAPMetric(map_value=map_value)
    fp_metric.update(output=[fp_res], target=[target_annotation])

    for item in fp_res[0, 0, ::]:
        _, _, conf, xmin, xmax, ymin, ymax = item
        if conf > confidence_threshold:
            total_image = draw_boxes_on_image([xmin, xmax, ymin, ymax], image, "red")

    axs[i, 1].imshow(total_image)

    int8_res = int8_compiled_model([input_image])[output_layer_int8]
    int8_metric = MAPMetric(map_value=map_value)
    int8_metric.update(output=[int8_res], target=[target_annotation])

    for item in int8_res[0, 0, ::]:
        _, _, conf, xmin, xmax, ymin, ymax = item
        if conf > confidence_threshold:
            total_image = draw_boxes_on_image(
                [xmin, xmax, ymin, ymax], total_image, "yellow"
            )
            int8_image = draw_boxes_on_image(
                [xmin, xmax, ymin, ymax], orig_image, "yellow"
            )

    axs[i, 2].imshow(int8_image)

    # Annotation
    for annotation in target_annotation:
        total_image = draw_boxes_on_image(annotation["bbox"], total_image, "green")

    axs[i, 0].imshow(image)
    axs[i, 0].set_title(Path(metadata["filename"]).stem)
    axs[i, 1].set_title(f"FP32 mAP: {fp_metric.avg_value[map_value]:.3f}")
    axs[i, 2].set_title(f"INT8 mAP: {int8_metric.avg_value[map_value]:.3f}")
    fig.suptitle(
        "Annotated (green) and detected boxes on FP (red) and INT8 (yellow) model"
    )

original_model_size = Path(ir_path).with_suffix(".bin").stat().st_size / 1024
quantized_model_size = (
    Path(compressed_model_path).with_suffix(".bin").stat().st_size / 1024
)

print(f"FP32 model size: {original_model_size:.2f} KB")
print(f"INT8 model size: {quantized_model_size:.2f} KB")