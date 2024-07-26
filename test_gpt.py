import itk

# Load the fixed and moving images
fixed_image_path = './Data/case6_gre1.nrrd'
moving_image_path = './Data/case6_gre2.nrrd'

fixed_image = itk.imread(fixed_image_path, itk.F)
moving_image = itk.imread(moving_image_path, itk.F)

# Define the registration components
Dimension = 3
FixedImageType = itk.Image[itk.F, Dimension]
MovingImageType = itk.Image[itk.F, Dimension]

TransformType = itk.TranslationTransform[itk.D, Dimension]
initial_transform = TransformType.New()

OptimizerType = itk.RegularStepGradientDescentOptimizerv4[itk.D]
optimizer = OptimizerType.New()

MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType]
metric = MetricType.New()

RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType]
registration = RegistrationType.New()

# Set up the registration
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)
registration.SetInitialTransform(initial_transform)
registration.SetFixedImage(fixed_image)
registration.SetMovingImage(moving_image)

# Define the optimizer parameters
optimizer.SetLearningRate(4.0)
optimizer.SetMinimumStepLength(0.001)
optimizer.SetNumberOfIterations(200)

# Start the registration process
registration.Update()

# Get the resulting transformation
final_transform = registration.GetTransform()

# Resample the moving image to align with the fixed image
ResampleFilterType = itk.ResampleImageFilter[MovingImageType, FixedImageType]
resampler = ResampleFilterType.New()
resampler.SetTransform(final_transform)
resampler.SetInput(moving_image)
resampler.SetSize(fixed_image.GetLargestPossibleRegion().GetSize())
resampler.SetOutputOrigin(fixed_image.GetOrigin())
resampler.SetOutputSpacing(fixed_image.GetSpacing())
resampler.SetOutputDirection(fixed_image.GetDirection())
resampler.SetDefaultPixelValue(0)

# Save the registered image
output_image_path = './Data/registered_image.nrrd'
itk.imwrite(resampler.GetOutput(), output_image_path)

print("Registration completed successfully.")

####
OutputPixelType = itk.F # was unsigned char in example, we would want in unsigned short
OutputImageType = itk.Image[OutputPixelType, Dimension]
DifferenceFilterType = itk.SquaredDifferenceImageFilter[FixedImageType, FixedImageType, OutputImageType]
difference = DifferenceFilterType.New()

WriterType = itk.ImageFileWriter[OutputImageType]
writer2 = WriterType.New()
writer2.SetInput(difference.GetOutput())

diff_fixed_resampled_filepath = "./Data/diff_fixed_resampled.nrrd"

difference.SetInput1(fixed_image)
difference.SetInput2(resampler.GetOutput())

writer2.SetFileName(diff_fixed_resampled_filepath)
writer2.Update()
####
# diff_fixed_moving_filepath = "./Data/diff_fixed_moving.nrrd"

# difference.SetInput1(fixed_image)
# difference.SetInput2(moving_image)

# writer2.SetFileName(diff_fixed_moving_filepath)
# writer2.Update()
####
# VectorPixelType = itk.Vector[itk.F, Dimension]
# DisplacementFieldImageType = itk.Image[VectorPixelType, Dimension]

# CoordinateRepType = itk.D # itk.ctype("double")
# DisplacementFieldGeneratorType = itk.TransformToDisplacementFieldFilter[DisplacementFieldImageType, CoordinateRepType]

# dispfieldGenerator = DisplacementFieldGeneratorType.New()
# dispfieldGenerator.UseReferenceImageOn()
# dispfieldGenerator.SetReferenceImage(fixed_image)
# dispfieldGenerator.SetTransform(resampler.GetOutput())

# dispfieldGenerator.Update()
####
# deformation_field_filepath = "./Data/deformation_field.nrrd" # should not be .nrrd ?

# FieldWriterType = itk.ImageFileWriter[DisplacementFieldImageType]
# fieldWriter = FieldWriterType.New()

# fieldWriter.SetInput(dispfieldGenerator.GetOutput())

# fieldWriter.SetFileName(deformation_field_filepath);
# fieldWriter.Update()