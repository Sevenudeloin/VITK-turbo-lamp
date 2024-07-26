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

MetricType = itk.MeanSquaresImageToImageMetricv4[FixedImageType, MovingImageType]
metric = MetricType.New()

OptimizerType = itk.LBFGSBOptimizerv4
optimizer = OptimizerType.New()

# Setting bounds for the optimizer
param_scaling = 1.0
number_of_parameters = transform.GetNumberOfParameters()

lower_bounds = [-param_scaling] * number_of_parameters
upper_bounds = [param_scaling] * number_of_parameters
bound_selection = [0] * number_of_parameters

optimizer.SetLowerBound(lower_bounds)
optimizer.SetUpperBound(upper_bounds)
optimizer.SetBoundSelection(bound_selection)

# Other optimizer parameters
optimizer.SetGradientConvergenceTolerance(1e-35)
optimizer.SetNumberOfIterations(500)
optimizer.SetMaximumNumberOfFunctionEvaluations(500)

RegistrationType = itk.ImageRegistrationMethodv4[FixedImageType, MovingImageType]
registration = RegistrationType.New()
registration.SetMetric(metric)
registration.SetOptimizer(optimizer)
registration.SetFixedImage(fixed_image)
registration.SetMovingImage(moving_image)
registration.SetInitialTransform(transform)

# Set up multi-resolution framework
number_of_levels = 3
shrink_factors_per_level = [4, 2, 1]
smoothing_sigmas_per_level = [2, 1, 0]

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