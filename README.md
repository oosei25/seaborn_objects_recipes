# Seaborn Objects Recipes

[![PyPI](https://img.shields.io/pypi/v/seaborn_objects_recipes.svg)](https://pypi.org/project/seaborn_objects_recipes/)
![Python versions](https://img.shields.io/pypi/pyversions/seaborn_objects_recipes.svg)
[![CI](https://github.com/oosei25/seaborn_objects_recipes/actions/workflows/actions.yml/badge.svg)](https://github.com/oosei25/seaborn_objects_recipes/actions)
[![Docs](https://img.shields.io/badge/docs-stable-blue)](https://oosei25.github.io/seaborn_objects_recipes/)

## About

seaborn_objects_recipes is a Python package that extends the functionality of the Seaborn library, providing custom recipes for enhanced data visualization. This package includes below features to augment your Seaborn plots with additional capabilities.

- [Rolling](https://github.com/Ofosu-Osei/seaborn_objects_recipes/blob/main/seaborn_objects_recipes/recipes/rolling.py)
- [LineLabel](https://github.com/Ofosu-Osei/seaborn_objects_recipes/blob/main/seaborn_objects_recipes/recipes/line_label.py)
- [Lowess](https://github.com/Ofosu-Osei/seaborn_objects_recipes/blob/main/seaborn_objects_recipes/recipes/lowess.py)
- [PolyFitWithCI](https://github.com/Ofosu-Osei/seaborn_objects_recipes/blob/main/seaborn_objects_recipes/recipes/plotting.py)

## Installation

To install `seaborn_objects_recipes`, run the following command:

```python
pip install seaborn_objects_recipes

```

## Example Gallery preview command for local testing:

```bash
    make -C docs html && open docs/_build/html/index.html
```

> For the full gallery and API, see the docs: [API-Gallery Docs](https://oosei25.github.io/seaborn_objects_recipes/)

## Contact

For questions or feedback regarding `seaborn_objects_recipes`, please contact [Ofosu Osei](mailto:goofosuosei@gmail.com).

[def]: https://github.com/Ofosu-Osei/seaborn_objects_recipes/actions/workflows/actions.yml

## Credits

* Special thanks to [@nickeubank](https://github.com/nickeubank) for the support and mentorship on this project

* Special thanks to [@JesseFarebro](https://github.com/JesseFarebro) for [Rolling, LineLabel](https://github.com/mwaskom/seaborn/discussions/3133)

* Special thanks to [@tbpassin](https://github.com/tbpassin) and [@kcarnold](https://github.com/kcarnold) for [LOWESS Smoother](https://github.com/mwaskom/seaborn/issues/3320)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

• ✅ Distinct x-value count valid for LOWESS (frac ≥ 2/n)

• ✅ CI columns present (ymin, ymax) when bootstrapping is on

• ✅ alpha respected and bootstraps defaulted if unset
