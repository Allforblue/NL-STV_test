import os
import pandas as pd
import geopandas as gpd
from typing import Union, Dict, Optional, Type
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义通用返回类型
DataType = Union[pd.DataFrame, gpd.GeoDataFrame]


class BaseLoader:
    """所有数据加载器的基类"""

    def load(self, file_path: str, **kwargs) -> DataType:
        """加载完整数据"""
        raise NotImplementedError("Must implement load method")

    def peek(self, file_path: str, n: int = 5) -> DataType:
        """
        仅加载前 n 行用于元数据分析 (AI 语义推断用)。
        默认实现是加载全量取头部，子类应重写以优化性能。
        """
        df = self.load(file_path)
        return df.head(n)
    def count_rows(self, file_path: str) -> int:
        """高效计算文件总行数"""
        # 默认实现：如果子类没优化，就加载全量计算（最慢，但安全）
        return len(self.load(file_path))


class CsvLoader(BaseLoader):
    """处理 .csv, .txt 文件"""

    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        try:
            # 自动推断分隔符（简单的嗅探逻辑）
            # 实际生产中可以使用 csv.Sniffer
            return pd.read_csv(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load CSV {file_path}: {e}")
            raise

    def peek(self, file_path: str, n: int = 5) -> pd.DataFrame:
        # 优化：只读取前 n 行，避免大文件内存爆炸
        return pd.read_csv(file_path, nrows=n)

    def count_rows(self, file_path: str) -> int:
        # 优化：仅读取第一列来计数，极大减少内存消耗
        try:
            # 这里的 usecols=[0] 确保我们只读一列
            df = pd.read_csv(file_path, usecols=[0])
            return len(df)
        except Exception:
            return len(pd.read_csv(file_path))


class ExcelLoader(BaseLoader):
    """处理 .xlsx, .xls 文件"""

    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_excel(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load Excel {file_path}: {e}")
            raise

    def peek(self, file_path: str, n: int = 5) -> pd.DataFrame:
        return pd.read_excel(file_path, nrows=n)


class ParquetLoader(BaseLoader):
    """处理 .parquet 文件 (高性能)"""

    def load(self, file_path: str, **kwargs) -> pd.DataFrame:
        try:
            return pd.read_parquet(file_path, **kwargs)
        except Exception as e:
            logger.error(f"Failed to load Parquet {file_path}: {e}")
            raise

    # Parquet 格式支持通过 pyarrow 快速读取 schema，这里暂时用 pandas 接口
    def peek(self, file_path: str, n: int = 5) -> pd.DataFrame:
        # read_parquet 不支持 nrows，但 Parquet 本身是列式存储，读取通常很快
        # 如果文件巨大，建议使用 pyarrow.parquet.read_table(limit=n) 优化
        df = pd.read_parquet(file_path)
        return df.head(n)

    def count_rows(self, file_path: str) -> int:
        # Parquet 格式在元数据里存了行数，读取速度极快，无需读取内容
        try:
            import pyarrow.parquet as pq
            metadata = pq.read_metadata(file_path)
            return metadata.num_rows
        except ImportError:
            # 如果没装 pyarrow，回退到 pandas 读取
            return len(pd.read_parquet(file_path, columns=[]))


class ShapefileLoader(BaseLoader):
    """处理 .shp, .geojson, .gpkg 等地理矢量数据"""

    def load(self, file_path: str, **kwargs) -> gpd.GeoDataFrame:
        try:
            gdf = gpd.read_file(file_path, **kwargs)

            # 检查坐标系，如果缺失则发出警告
            if gdf.crs is None:
                logger.warning(f"⚠️ Warning: {Path(file_path).name} has NO projection (CRS).")
            else:
                logger.info(f"Loaded geospatial data with CRS: {gdf.crs.to_string()}")

            return gdf
        except Exception as e:
            logger.error(f"Failed to load Spatial Data {file_path}: {e}")
            raise

    def peek(self, file_path: str, n: int = 5) -> gpd.GeoDataFrame:
        # gpd.read_file 支持 rows 参数 (Geopandas >= 0.11.0)
        try:
            return gpd.read_file(file_path, rows=n)
        except TypeError:
            # 兼容旧版本
            return gpd.read_file(file_path).head(n)

    def count_rows(self, file_path: str) -> int:
        # GeoPandas 读取需要一点时间，但通常 Shapefile 不会太大
        # 为了避免读取复杂的 geometry，我们可以只读属性表 (.dbf)
        # 但为了代码简单，这里暂时直接读取，后续可优化为只读 dbf
        return len(gpd.read_file(file_path, ignore_geometry=True))


class LoaderFactory:
    """工厂类：根据文件扩展名分发加载器"""

    _loaders: Dict[str, Type[BaseLoader]] = {
        '.csv': CsvLoader,
        '.txt': CsvLoader,
        '.xlsx': ExcelLoader,
        '.xls': ExcelLoader,
        '.parquet': ParquetLoader,
        '.shp': ShapefileLoader,
        '.geojson': ShapefileLoader,
        '.gpkg': ShapefileLoader,
        '.kml': ShapefileLoader
    }

    @classmethod
    def get_loader(cls, file_path: str) -> BaseLoader:
        """根据文件路径获取对应的加载器实例"""
        ext = Path(file_path).suffix.lower()

        loader_class = cls._loaders.get(ext)
        if not loader_class:
            # 默认为 CSV 加载器，或者抛出不支持的异常
            logger.warning(f"Unknown extension '{ext}', defaulting to CSV loader.")
            return CsvLoader()

        return loader_class()


# --- 单元测试/使用示例 ---
if __name__ == "__main__":
    # 创建一些假文件用于测试 (为了演示，实际运行需保证文件存在)
    # 请手动在 tests/data 目录下放入真实的 yellow_tripdata_sample.parquet 或 .csv

    test_files = [
        "../../data/taxi_zone_lookup.csv",  # 假设文件路径
        "../../data/yellow_tripdata_2025-01.parquet",
        "../../data/taxi_zones.shp"  # 只需要指向 .shp，需同目录下有 .dbf, .shx
    ]

    print("=== Testing Loader Factory ===")

    for f_path in test_files:
        if not os.path.exists(f_path):
            print(f"Skipping {f_path} (File not found)")
            continue

        print(f"\nProcessing: {f_path}")
        try:
            # 1. 获取加载器
            loader = LoaderFactory.get_loader(f_path)

            # 2. Peek 模式 (只读 3 行，极速)
            df_preview = loader.peek(f_path, n=3)

            print(f"Type: {type(df_preview)}")
            print(f"Columns: {list(df_preview.columns)}")
            print("Preview Data:")
            print(df_preview.iloc[0].to_dict())  # 打印第一行数据

            # 如果是地理数据，打印 CRS
            if isinstance(df_preview, gpd.GeoDataFrame):
                print(f"CRS: {df_preview.crs}")

        except Exception as e:
            print(f"Error: {e}")