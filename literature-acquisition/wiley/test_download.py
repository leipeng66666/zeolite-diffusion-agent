from wiley_downloader import WileyDownloader

# 创建下载器
downloader = WileyDownloader(output_dir="test_downloads")

# 测试下载少量文献
print("开始测试下载...")
downloader.batch_download(
    keyword="machine learning",
    start=1,
    end=2,
    format_type="pdf"
)
