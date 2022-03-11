
extracted_face = extract_face_from_image('iacocca_1.jpg')

def get_model_scores(faces):
  samples = asarray(faces, 'float32')

  # prepare the data for the model
  samples = preprocess_input(samples, version=2)

  # create a vggface model object
  model = VGGFace(model='resnet50',
      include_top=False,
      input_shape=(224, 224, 3),
      pooling='avg')

  # perform prediction
  return model.predict(samples)

faces = [extract_face_from_image(image_path)
         for image_path in ['iacocca_1.jpg', 'iacocca_2.jpg']]

model_scores = get_model_scores(faces)


if cosine(model_scores[0], model_scores[1]) <= 0.4:
  print("Faces Matched")

store_image('https://cdn.vox-cdn.com/thumbor/Ua2BXGAhneJHLQmLvj-ZzILK-Xs=/0x0:4872x3160/1820x1213/filters:focal(1877x860:2655x1638):format(webp)/cdn.vox-cdn.com/uploads/chorus_image/image/63613936/1143553317.jpg.5.jpg',
            'chelsea_1.jpg')

image = plt.imread('chelsea_1.jpg')
faces_staring_xi = detector.detect_faces(image)

highlight_faces('chelsea_1.jpg', faces_staring_xi)

store_image('https://cdn.vox-cdn.com/thumbor/mT3JHQtZIyInU8_uGxVH-TCbF50=/0x415:5000x2794/1820x1213/filters:focal(1878x1176:2678x1976):format(webp)/cdn.vox-cdn.com/uploads/chorus_image/image/65171515/1161847141.jpg.0.jpg',
            'chelsea_2.jpg')

image = plt.imread('chelsea_2.jpg')
faces = detector.detect_faces(image)

highlight_faces('chelsea_2.jpg', faces)

slavia_faces = extract_face_from_image('chelsea_1.jpg')
liverpool_faces = extract_face_from_image('chelsea_2.jpg')

model_scores_starting_xi_slavia = get_model_scores(slavia_faces)
model_scores_starting_xi_liverpool = get_model_scores(liverpool_faces)
``
for idx, face_score_1 in enumerate(model_scores_starting_xi_slavia):
  for idy, face_score_2 in enumerate(model_scores_starting_xi_liverpool):
    score = cosine(face_score_1, face_score_2)
    if score <= 0.4:
      # Printing the IDs of faces and score
      print(idx, idy, score)
      # Displaying each matched pair of faces
      plt.imshow(slavia_faces[idx])
      plt.show()
      plt.imshow(liverpool_faces[idy])
      plt.show()
