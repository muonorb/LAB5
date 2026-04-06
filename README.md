[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-2972f46106e565e64193e422d61a12cf1da4916b45550586e14ef0a7c637dd04.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=23373422)
# Image Mosaic Generator: Build, Profile, and Optimize
![mosaic.png](mosaic.png)
## 📋 Overview

In this assignment, you will build an Interactive Image Mosaic Generator that reconstructs an input image using small tiles (mini-images), then systematically profile and optimize your implementation for performance and maintainability.

**Core Philosophy**: First make it work, then make it fast, then make it maintainable.

This project is divided into two parts:
- **Part 1**: Build a working mosaic generator with a Gradio interface
- **Part 2**: Profile, optimize, and refactor your implementation into production-quality code

---

## 🎯 Learning Objectives

By completing this project, you will:

- Implement image processing algorithms for mosaic generation
- Build interactive web interfaces using Gradio
- Apply profiling tools (cProfile, line_profiler) to identify bottlenecks
- Optimize computational operations using NumPy vectorization
- Refactor monolithic code into modular, reusable components
- Measure and verify performance improvements scientifically
- Write production-quality, maintainable code with proper documentation

---

## 📝 Part 1: Building the Image Mosaic Generator

### Overview

Build a working image mosaic generator that divides an input image into a grid and replaces each grid section with an appropriate tile from a predefined tile set.

### Implementation Steps

#### Step 1: Image Selection and Preprocessing

- Choose a set of test images
- Optional: apply [color quantization](https://en.wikipedia.org/wiki/Color_quantization) to simplify color variations
- Resize the image to a fixed resolution to ensure consistency (it is OK to slightly crop images so that a grid can be applied to them)

#### Step 2: Image Grid and Thresholding

- Divide the image into a fixed-size grid (e.g., 32×32 tiles)
- Use image processing techniques to analyze the intensity or color values in each grid cell
- Implement your grid operations using vectorized NumPy operations instead of nested loops for better performance
- Classify each cell into a color category

#### Step 3: Tile Mapping

- Prepare a set of image tiles
- Replace each grid cell with a corresponding tile based on the classified intensity/color

#### Step 4: Building the Gradio Interface

- Install [Gradio](https://gradio.app/) in your Python environment
- Implement a function that:
  - Accepts an image
  - Applies grid division
  - Reconstructs the image using image tiles
  - Outputs the final mosaic-style image
- Design a simple interface that allows users to:
  - Upload an image
  - Adjust parameters such as grid size or tile set
  - View the final mosaic-style reconstruction

#### Step 5: Performance Metric

Use a similarity metric to compare the original image and the reconstructed mosaic.

Some possibilities:

- Mean Squared Error (MSE) between the original and mosaic image
- Structural Similarity Index (SSIM) to quantify visual similarity

#### Step 6: Initial Performance Analysis

Analyze computational performance:

- Measure and report processing time for different grid sizes (e.g., 16×16, 32×32, 64×64)
- Include a brief analysis of how performance scales with grid size
- Compare execution time between vectorized and loop-based implementations

### Part 1 Guidelines

- **Creative Freedom**: Experiment with different tile designs, image processing methods, and grid sizes
- **Presentation**: Show the original image, the segmented image, and the final mosaic result
- **Focus on Functionality**: Get it working correctly first; optimization comes in Part 2

---

## Gradio Tips

- When testing your Gradio app locally at `http://localhost:7860`, use the command `gradio app.py` instead of `python app.py` in the terminal. This will automatically refresh the app as you make changes
- You can use the `examples` parameter in `gr.Interface(···)` to provide sample inputs (e.g., testing images). These examples will appear below the UI components and can be clicked to populate the interface, making it faster to test your app
- For deployment, use Hugging Face Spaces for a permanent link (do not use the 3-day temporary link)
- Using Hugging Face Spaces is similar to using GitHub. For more details:
  - [https://www.gradio.app/guides/sharing-your-app](https://www.gradio.app/guides/sharing-your-app)
  - [https://huggingface.co/blog/gradio-spaces](https://huggingface.co/blog/gradio-spaces)
- Include specific library versions in `requirements.txt` (e.g., `opencv-python==4.9.0.80`) to ensure consistent results across environments

---

## ⚡ Part 2: Profile, Optimize, and Refactor

### Overview

Transform your working implementation into a high-performance, production-quality system through systematic profiling, optimization, and refactoring.

### Part 2.1: Profiling Your Implementation

#### Starting Point

Use your Part 1 code as the baseline.

#### Tasks

**1. Set up profiling infrastructure**

- Install line_profiler: `pip install line_profiler`
- Create a test suite with images of different sizes (e.g., 256×256, 512×512, 1024×1024)
- Test with different grid sizes (e.g., 16×16, 32×32, 64×64)

**2. Measure baseline performance**

- Time the complete mosaic generation process
- Record times for each image size and grid size combination
- Identify which operations take the most time

**3. Use cProfile for function-level analysis**

- Profile your main mosaic generation function
- Identify which functions consume the most time
- Look for unexpected bottlenecks (e.g., repeated file I/O, redundant calculations)

**4. Use line_profiler for detailed analysis**

- Profile your slowest functions line-by-line
- Identify specific lines that are performance bottlenecks
- Look for nested loops, repeated operations, or inefficient data structures

**Deliverable**: Create a Jupyter notebook called `profiling_analysis.ipynb` that documents your profiling process, shows cProfile and line_profiler outputs, and identifies at least 3 specific bottlenecks to address.

---

### Part 2.2: Optimize for Performance

#### Common Bottlenecks and Solutions

**🔴 Bottleneck #1: Grid Operations with Nested Loops**

Problem: Using nested loops to divide image into grid cells
```python
for i in range(grid_height):
    for j in range(grid_width):
        cell = image[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
        # process cell...
```

Solution: Use NumPy array reshaping and vectorized operations
```python
cells = image.reshape(grid_h, cell_h, grid_w, cell_w, 3)
cells = cells.transpose(0, 2, 1, 3, 4)
cells = cells.reshape(-1, cell_h, cell_w, 3)
```

**🔴 Bottleneck #2: Color Matching with Loops**

Problem: Comparing each cell to each tile individually

Solution: Pre-compute tile features (average color, dominant color) and use vectorized distance calculations

**🔴 Bottleneck #3: Repeated Image Loading**

Problem: Loading tiles from disk multiple times

Solution: Load all tiles once at startup and cache in memory

#### Optimization Tasks

**1. Vectorize grid operations**

- Replace all nested loops with NumPy array operations
- Use array slicing, reshaping, and broadcasting
- Ensure no pixel-by-pixel operations

**2. Optimize color matching**

- Pre-compute representative colors for all tiles
- Use vectorized distance calculations (e.g., `np.linalg.norm`)
- Consider using k-d tree or similar structure for fast nearest neighbor lookup

**3. Implement caching**

- Cache loaded tiles in memory
- Cache computed tile features
- Avoid redundant calculations

**4. Optimize image resizing operations**

- Use efficient interpolation methods
- Resize once at the beginning, not repeatedly

**5. Profile again and verify improvements**

- Measure performance with the same test cases
- Calculate speedup factors for each optimization
- Ensure correctness is maintained (output should be identical or very similar)

#### Performance Target

Your optimized version should be at least **20× faster** than your original Part 1 code for 512×512 images with 32×32 grids. Aim for **50-100× speedup**!

**Deliverable**: Update your code with optimizations and create a section in your notebook showing before/after performance comparisons with timing data and speedup calculations.

---

### Part 2.3: Refactor for Modularity

#### Goal

Transform your single-file script into a well-organized, maintainable package.

#### Required Module Structure
```
mosaic_generator/
├── __init__.py
├── image_processor.py      # Image loading, resizing, grid division
├── tile_manager.py         # Tile loading, caching, feature extraction
├── mosaic_builder.py       # Main mosaic construction logic
├── metrics.py              # Similarity metrics (MSE, SSIM)
├── config.py               # Configuration constants
└── utils.py                # Helper functions

app.py                      # Gradio interface
requirements.txt            # Dependencies with versions
README.md                   # Documentation
profiling_analysis.ipynb    # Profiling notebook from Part 2.1
tests/                      # Unit tests (optional but recommended)
└── test_mosaic.py
```

#### Refactoring Tasks

**1. Separate concerns into modules**

- **image_processor.py**: Functions for image loading, preprocessing, grid creation
- **tile_manager.py**: TileManager class that handles loading, caching, and feature extraction
- **mosaic_builder.py**: MosaicBuilder class with the main reconstruction algorithm
- **metrics.py**: Functions to compute similarity metrics
- **config.py**: Constants like default grid size, tile dimensions, etc.

**2. Add proper error handling**

- Validate input parameters (image size, grid size)
- Handle missing tiles gracefully
- Provide informative error messages

**3. Write comprehensive docstrings**

Every function should have a docstring with:

- Brief description
- Parameters with types
- Return value with type
- Example usage (optional)

**4. Update Gradio interface**

- Keep the same functionality from Part 1
- Add performance metrics display (processing time)
- Use the new modular code structure
- Ensure the interface is still intuitive

**5. Create a comprehensive README**

- Installation instructions
- Usage examples
- Performance benchmarks comparing Part 1 to Part 2
- Link to deployed Gradio app

#### Example Module Usage
```python
from mosaic_generator import MosaicBuilder, TileManager

# Initialize
tile_manager = TileManager(tile_directory='tiles/')
builder = MosaicBuilder(tile_manager, grid_size=(32, 32))

# Generate mosaic
mosaic = builder.create_mosaic(input_image)
similarity = builder.compute_similarity(input_image, mosaic)
```

**Deliverable**: Refactored codebase with proper module structure, updated Gradio app, and comprehensive README.

---

## 📊 Performance Report Requirements

Create a 2-3 page report (PDF) that includes:

### 1. Executive Summary

- Brief overview of your implementation approach
- Key optimizations performed
- Performance improvements achieved

### 2. Part 1 Implementation

- Description of your mosaic generation algorithm
- Tile selection strategy
- Initial performance characteristics

### 3. Profiling Analysis (Part 2.1)

- Screenshots or excerpts of cProfile output
- Identification of top 3 bottlenecks
- Explanation of why these were bottlenecks

### 4. Optimization Strategy (Part 2.2)

- Description of each optimization applied
- Technical explanation of why each optimization works
- Code snippets showing before/after (optional)

### 5. Performance Results

Table showing timing results for different configurations:

| Image Size | Grid Size | Part 1 Time (s) | Part 2 Time (s) | Speedup |
|------------|-----------|-----------------|-----------------|---------|
| 256×256    | 16×16     | ...             | ...             | ...     |
| 512×512    | 32×32     | ...             | ...             | ...     |
| 1024×1024  | 64×64     | ...             | ...             | ...     |

Graphs showing:

- Execution time comparison (bar chart)
- Speedup factors (bar chart)
- Scaling behavior with image/grid size (line plot)

### 6. Code Quality Improvements (Part 2.3)

- Description of refactoring performed
- Benefits of modular structure
- Any design patterns used

### 7. Challenges and Lessons Learned

- Challenges encountered during implementation and optimization
- Trade-offs made (if any)
- Key takeaways from the project

---

## 📤 Submission Requirements

Submit everything on GitHub:

### 1. Source Code

- All source code with modular structure (from Part 2.3)
- `profiling_analysis.ipynb` with profiling results (from Part 2.1)
- `requirements.txt` with specific versions
- `README.md` with installation and usage instructions
- Updated Gradio app (`app.py`)
- Sample test images (or links to download them)

### 2. Performance Report (PDF)

2-3 page document following the structure outlined above

### 3. Live Gradio Demo

Link to your deployed app on Hugging Face Spaces

- Must be functional and accessible
- Should display processing time for transparency
- Include example images for testing

### 4. Demo Video (Optional - Bonus +5 points)

2-3 minute video demonstrating:

- Your working mosaic generator
- Original Part 1 performance vs optimized Part 2 performance
- Side-by-side timing comparison
- Brief explanation of key optimizations
- Walkthrough of your Gradio interface

---

## 🎯 Evaluation Criteria

- **Correctness (15 points)**: Is the image-to-mosaic conversion implemented correctly?
- **Creativity (10 points)**: Are different tile types or unique mapping strategies used?
- **Interface (10 points)**: Is the Gradio interface intuitive and user-friendly?
- **Metrics (5 points)**: Does the similarity metric correctly evaluate results?
- **Profiling Analysis (15 points)**: Thorough identification and documentation of bottlenecks
- **Performance Improvement (20 points)**: Achieves significant speedup (20×+ for full credit)
- **Code Quality (15 points)**: Modular structure, documentation, error handling
- **Report Quality (10 points)**: Clear explanation of methods, results, and insights

---

## 💡 Tips for Success

### General Tips

- **Start early**: Both parts require significant work
- **Test incrementally**: Make sure each component works before moving on
- **Document as you go**: Don't leave documentation for the end
- **Keep backups**: Commit/Push early and often

### Part 1 Tips

- Focus on getting something working first
- Don't over-optimize initially—that's what Part 2 is for
- Test with multiple images to ensure robustness
- Make the Gradio interface simple and intuitive

### Part 2 Optimization Tips

- **Start with profiling**: Don't optimize blindly! Let the profiler guide you
- **One optimization at a time**: Make changes incrementally and measure after each change
- **Verify correctness**: After each optimization, ensure output is still correct
- **Focus on the hot path**: Optimize the code that runs most frequently first

### Part 2 Refactoring Tips

- **Single Responsibility Principle**: Each module should have one clear purpose
- **Don't Repeat Yourself (DRY)**: Extract repeated code into reusable functions
- **Keep the interface simple**: Hide complexity inside modules, expose simple APIs

### Common Pitfalls to Avoid

- ❌ Optimizing before profiling (premature optimization)
- ❌ Breaking functionality while optimizing (always verify correctness)
- ❌ Over-engineering the refactoring (keep it simple and practical)
- ❌ Forgetting to update Gradio app to use new modules
- ❌ Not testing with different image sizes
- ❌ Leaving hardcoded values instead of using configuration

---

## 📚 Resources

- Course Materials: Week 10 profiling tutorial and notes
- NumPy Documentation: https://numpy.org/doc/stable/
- Python Profiling: cProfile documentation
- Gradio Documentation: https://www.gradio.app/docs
- Hugging Face Spaces: https://huggingface.co/spaces
- Gradio Video Tutorial (see course materials)

---

## ❓ FAQ

**Q: My Part 1 code is already pretty fast. What should I do?**

A: There is always room for improvement. Focus heavily on the refactoring portion (Part 2.3) and consider adding new features like batch processing or memory optimization. You can also implement multiple optimization strategies and compare them.

**Q: How detailed should my profiling analysis be?**

A: Show clear before/after comparisons, identify specific bottlenecks with evidence from profiler output, and explain what you learned. Quality over quantity: focus on insights, not just raw data dumps.

**Q: What if my optimized code produces slightly different output?**

A: Small numerical differences due to floating-point operations are acceptable. The visual result should be essentially identical. If there are significant differences, something is wrong.

**Q: Can I add new features beyond the requirements?**

A: Yes! But complete the required components first. Additional features can include batch processing, different tile matching algorithms, advanced visual effects, or interactive parameter tuning.

**Q: Should I use object-oriented or functional programming?**

A: Both where they make sense.

---

## 🎯 Learning Outcomes

After completing this project, you will have:

✅ Practical experience with image processing and mosaic generation algorithms  
✅ Proficiency in building interactive web applications with Gradio  
✅ Hands-on experience profiling real code and identifying bottlenecks  
✅ Mastery of NumPy vectorization for performance optimization  
✅ Understanding of when and how to optimize code effectively  
✅ Skills in refactoring monolithic code into maintainable modules  
✅ Experience with professional software development practices  
✅ Ability to measure and communicate performance improvements  
✅ A portfolio piece showcasing implementation, optimization, and design skills

**Remember**: The goal isn't just to make code that works or code that's fast. It's to learn how to systematically build, analyze, optimize, and improve software while maintaining code quality. These skills are essential for any data scientist or AI engineer working with large-scale data.

---

Questions? Post on Piazza or attend office hours.
