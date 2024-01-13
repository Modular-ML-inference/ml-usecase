
import torch
import numpy as np

class Processor():

    def __init__(self, images, desired_size, device = 'cuda'):
        self.target_size = desired_size
        self.device = device
        self.images = []
        for image in images:
            img = self.prepare_img(image)
            img = img.to(self.device)
            self.images.append(img)
        
    
    def resize(self, img):
        """
        Resizes the image to the target size given during dataset initialization

        Args:
            img: source image in PIL format

        Returns:
            img: resized image in PIL format
        """
        return img.resize(self.target_size)

    def img_to_tensor(self, img):
        """
        Converts image to torch.Tensor previously also scaling it from [0, 255] to [0, 1]

        Args:
            img: 3-channel image array

        Returns:
            torch.Tensor 
        """
        return torch.Tensor(img/255)

    def transpose_img(self, img):
        """
        Transposes the axis of the image from [H, W, C] to match tensor dimensions [C, H, W]

        Args:
            img: 3-channel image array

        Returns:
            img: transposed image array
        """
        return img.transpose((2, 0, 1))
    
    def from_PIL_to_array(self, img):
        """
        Converts image read by Image.open() to numpy array
        for future processing by opencv

        Args:
            img: PIL image

        Returns:
            img: image array
        """
        return np.array(img)

    def prepare_img(self, img):
        
        img = self.resize(img)
        img = self.from_PIL_to_array(img)
        img = self.transpose_img(img)
        img = self.img_to_tensor(img)
        
        return img
    
    def filter_outputs(self, outputs, conf_thresh):
        for i in range(len(outputs)):
            if len(outputs[i]["scores"]) > 0:
                scores = outputs[i]["scores"].detach().cpu().numpy()
                srt_det = np.flip(np.argsort(scores))
                srt_det = srt_det[np.where(scores > conf_thresh)[0]]

                outputs[i]['scores'] = outputs[i]["scores"][srt_det]
                outputs[i]["masks"] = outputs[i]["masks"][srt_det]
                outputs[i]["labels"] = outputs[i]["labels"][srt_det]
                outputs[i]["boxes"] = outputs[i]["boxes"][srt_det]
        return outputs

    def prepare_outputs(self, outputs):
        results = np.array([], dtype = np.int32)
        boxes = []
        labels = np.array([], dtype = np.int64)
        scores = np.array([], dtype = np.float32)
        masks = []
        for output in outputs:
            num_detections = len(output['scores'])
            results = np.append(results, num_detections)
            if num_detections > 0:
                [boxes.append(box) for box in output['boxes'].detach().to('cpu').numpy()]
                [masks.append(mask) for mask in output['masks'].detach().to('cpu').numpy()]
                labels = np.append(labels, output['labels'].detach().to('cpu').numpy())
                scores = np.append(scores, output['scores'].detach().to('cpu').numpy())
                
        boxes = np.array(boxes, dtype = np.float32)
        masks = np.array(masks, dtype = np.uint8)
        return {'results': results, 'labels': labels, 'boxes': boxes, 'scores': scores, 'masks': masks}
