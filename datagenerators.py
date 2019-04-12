import datasource
import cv2
import gc

import numpy

image_cache = dict()
seed = 1337


class DataGenerator:

    def __init__(self, image_size, data_root=None, splits_root=None):
        self.data = datasource.index_split_data(data_root, splits_root)
        self.classes, self.class_names = self.generate_classes()
        self.image_size = image_size
        DataGenerator.cache_data_set(self.data, self.image_size)

    def generate_classes(self):
        assert self.data is not None

        class_names = list([c.name for c in self.data])
        classes = dict(zip(class_names, numpy.eye(len(class_names))))

        return classes, class_names

    def generate_training_data(self, batch_size):
        assert self.classes is not None
        assert self.data is not None

        data_set = list([(self.classes[data_class.name], file) for data_class in
                         self.data for file in data_class.training_set])

        return self.generate_data(data_set, batch_size)

    def generate_testing_data(self, batch_size):
        assert self.classes is not None
        assert self.data is not None

        data_set = list([(self.classes[data_class.name], file) for data_class in
                         self.data for file in data_class.testing_set])

        return self.generate_data(data_set, batch_size)

    def generate_data(self, data, batch_size):
        assert data is not None
        assert batch_size is not 0

        global seed
        seed = seed + 1

        while True:
            # Ensure randomisation per epoch, but keep deterministic
            numpy.random.seed(seed=seed)
            numpy.random.shuffle(data)

            X = []
            Y = []

            # we removed count var from here, loop over range of images
            for j in range(len(data)):

                # [1] Call the load_images function and append the image in X.
                image = DataGenerator.load_image(data[j][1], self.image_size)
                label = data[j][0]
                X.append(image)
                # [2] Create a one-hot encoding with np.eye and append the one-hot vector to Y.
                Y.append(label)  # already receiving 1-hot from load_image

                # [3] Compare the count and batch_size (hint: modulo operation) and if so:
                # we use j+1 as count var was previously equivalent to this
                # when we hit a value of j+1 that is a multiple of batch size we run below block
                if ((j + 1) % batch_size) == 0:
                    #   - Use yield to return X,Y as numpy arrays with types 'float32' and 'uint8' respectively
                    X = numpy.array(X, dtype=numpy.float32)
                    X = X.reshape(batch_size, 200, 200, 1)
                    Y = numpy.array(Y, dtype=numpy.uint8)
                    yield X, Y

                    #   - delete X,Y
                    del X
                    del Y

                    #   - set X,Y to []
                    X = []
                    Y = []

                    # garbage collect
                    gc.collect()

    def get_label(self, one_hot):
        return self.classes[one_hot.index(max(one_hot))]

    def get_predictions(self, model, batch_size):
        for batch in self.generate_testing_data(batch_size):
            yield [(item_data, model.predict(item_data), item_class) for item_class, item_data in batch]

    # Cache all images
    @staticmethod
    def cache_data_set(data, image_size):
        assert data is not None
        assert image_size is not None
        assert image_size > 0

        all_data = []

        for data_class in data:
            all_data.extend(data_class.testing_set)
            all_data.extend(data_class.training_set)

        for path in all_data:
            DataGenerator.load_image(path, image_size)

    # Prepare and load images
    @staticmethod
    def load_image(image_path, image_size):

        global image_cache

        cache_key = str(image_path) + "x" + str(image_size)

        cached = image_cache.get(cache_key, None)
        if cached is not None:
            return cached

        # [2] Load the image in greyscale with OpenCV.
        # use cv2 image read function, and greyscale parameter
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # [3] Find the dimension that is the smallest between the height and the width and assign it to the crop_dim var
        # get values of height and width from image shape
        height, width = image.shape[:2]

        # create a tuple that flags 1 on the larger dimension, height in position 0 and width in position 1
        if height < width:
            crop_dim = (0, 1)
        else:
            crop_dim = (1, 0)

        # [4] Crop the centre of the image based on the crop_dim dimension for both the height and width.
        # use our crop_dim tuple in calculation to determine what the margin (distance from edge of image)
        margin = int(crop_dim[0] * (height - width) / 2 + crop_dim[1] * (width - height) / 2)
        # use the crop_dim and calculated margin to exclude the margin from the appropriate dimension
        image = image[(crop_dim[1] * margin):(width - crop_dim[1] * margin),
                (crop_dim[0] * margin):(height - crop_dim[0] * margin)]

        # [5] Resize the image to 48 x 48 and divide it with 255.0 to normalise it to floating point format.
        # use OpenCV resize function
        image = cv2.resize(image, (image_size, image_size))

        # set the image as a float type and divide by 255.0 to normalize
        image.astype(float)
        image = image / 255.0

        # Cache image for quicker loading next time
        image_cache[cache_key] = image

        return image
