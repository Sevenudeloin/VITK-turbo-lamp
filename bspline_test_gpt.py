import itk

# Step 1: Load the two NRRD images
fixed_image_path = "./Data/case6_gre1.nrrd"
moving_image_path = "./Data/case6_gre2.nrrd"

fixed_image = itk.imread(fixed_image_path, itk.F)
moving_image = itk.imread(moving_image_path, itk.F)

# Step 2: Set up the B-spline registration method
Dimension = 3
FixedImageType = itk.Image[itk.F, Dimension]
MovingImageType = itk.Image[itk.F, Dimension]

TransformType = itk.BSplineTransform[itk.D, Dimension, 3]
InitializerType = itk.BSplineTransformInitializer[TransformType, FixedImageType]
transform = TransformType.New()
initializer = InitializerType.New(Transform=transform, Image=fixed_image)
initializer.InitializeTransform()

# Finer B-spline grid resolution
grid_size = [8, 8, 8]  # Adjust grid size based on your images
transform.SetTransformDomainMeshSize(grid_size)

MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType]
metric = MetricType.New()

OptimizerType = itk.LBFGSBOptimizerv4
optimizer = OptimizerType.New()

# Setting bounds for the optimizer
param_scaling = 1.0
number_of_parameters = transform.GetNumberOfParameters()

boundSelect = itk.Array[itk.D](number_of_parameters)
upperBound = itk.Array[itk.D](number_of_parameters)
lowerBound = itk.Array[itk.D](number_of_parameters)

UNBOUNDED = 0.0
boundSelect.Fill(UNBOUNDED)
upperBound.Fill(0.0)
lowerBound.Fill(0.0)

optimizer.SetBoundSelection(boundSelect)
optimizer.SetUpperBound(upperBound)
optimizer.SetLowerBound(lowerBound)

optimizer.SetCostFunctionConvergenceFactor(1.e7)
optimizer.SetGradientConvergenceTolerance(1e-6)
optimizer.SetNumberOfIterations(200)
optimizer.SetMaximumNumberOfFunctionEvaluations(30)
optimizer.SetMaximumNumberOfCorrections(5)

RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType]
registration = RegistrationType.New()
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)
registration.SetFixedImage(fixed_image)
registration.SetMovingImage(moving_image)
registration.SetInitialTransform(transform)

# Update multi-resolution framework for finer registration
number_of_levels = 4  # Increase number of levels
shrink_factors_per_level = [8, 4, 2, 1]
smoothing_sigmas_per_level = [4, 2, 1, 0]
# number_of_levels = 3
# shrink_factors_per_level = [4, 2, 1]
# smoothing_sigmas_per_level = [2, 1, 0]

registration.SetNumberOfLevels(number_of_levels)
registration.SetShrinkFactorsPerLevel(shrink_factors_per_level)
registration.SetSmoothingSigmasPerLevel(smoothing_sigmas_per_level)

# Step 3: Execute the registration
registration.Update()

# Apply the transform to the moving image
resampler = itk.ResampleImageFilter.New(Input=moving_image, Transform=registration.GetTransform(), UseReferenceImage=True)
resampler.SetReferenceImage(fixed_image)
resampler.Update()

resampled_image = resampler.GetOutput()

# Save the resampled image
output_image_path = "./Data/register_image.nrrd"
itk.imwrite(resampled_image, output_image_path)

print(f"Registration complete. Output saved to {output_image_path}")

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
VectorPixelType = itk.Vector[itk.F, Dimension]
DisplacementFieldImageType = itk.Image[VectorPixelType, Dimension]

CoordinateRepType = itk.D # itk.ctype("double")
DisplacementFieldGeneratorType = itk.TransformToDisplacementFieldFilter[DisplacementFieldImageType, CoordinateRepType]

dispfieldGenerator = DisplacementFieldGeneratorType.New()
dispfieldGenerator.UseReferenceImageOn()
dispfieldGenerator.SetReferenceImage(fixed_image)
dispfieldGenerator.SetTransform(resampler.GetOutput())

dispfieldGenerator.Update()

deformation_field_filepath = "./Data/deformation_field.nrrd" # should not be .nrrd ?

FieldWriterType = itk.ImageFileWriter[DisplacementFieldImageType]
fieldWriter = FieldWriterType.New()

fieldWriter.SetInput(dispfieldGenerator.GetOutput())

fieldWriter.SetFileName(deformation_field_filepath)
fieldWriter.Update()