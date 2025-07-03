# HarvardX Capstone - MovieLens Project

This repository contains the final capstone project for the HarvardX Data Science Professional Certificate.

The objective is to develop a movie recommendation system using the [MovieLens 10M dataset](https://grouplens.org/datasets/movielens/10m/). The model will be evaluated using the Root Mean Square Error (RMSE) metric, following the guidelines provided in the course.

## Project Structure

```
├── src/
│ ├── data_preparation.R # Data cleaning and transformation
│ ├── movielens_project_code.R # Modeling and RMSE evaluation
│ └── MovieLens_Report.Rmd # Final report with explanations and results
├── ml-10M100K/ # Extracted MovieLens 10M dataset
├── LICENSE
├── README.md
└── .gitignore
```

## Requirements

This project uses R. The following libraries are required:
- `tidyverse`
- `caret`

You can install them using:

- `install.packages("tidyverse")`
- `install.packages("caret")`

## Getting Started

1. Clone the repository:

   ```
   git clone https://github.com/your-username/harvardx-capstone-movielens.git
   cd harvardx-capstone-movielens
   ```



2. Install the required R libraries:

```
install.packages("tidyverse")
install.packages("caret")
```

3. Download and extract the MovieLens 10M dataset into the root directory of the project. You should see a folder named ml-10M100K.

4. Run the R scripts in order:

```
src/data_preparation.R — for data cleaning and preparation

src/movielens_project_code.R — for model training and evaluation
```

5. Open src/MovieLens_Report.Rmd in RStudio to explore the report, or knit it to PDF.


## License

This project was developed as part of the HarvardX Data Science Professional Certificate (via edX). The code is original and adheres to the [edX Honor Code](https://www.edx.org/edx-honor-code) and [Terms of Service](https://www.edx.org/edx-terms-service).
