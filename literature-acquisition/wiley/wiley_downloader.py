import requests
from bs4 import BeautifulSoup
import os
import time
from urllib.parse import quote, urljoin
import json
from pathlib import Path


class WileyDownloader:
    def __init__(self, output_dir="downloads"):
        self.base_url = "https://onlinelibrary.wiley.com"
        self.search_url = f"{self.base_url}/action/doSearch"
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        })
        
    def search_articles(self, keyword, start=1, end=10):
        """搜索文献并返回文章列表"""
        print(f"正在搜索关键词: {keyword}")
        articles = []
        
        # 计算需要的页数（每页通常20篇）
        page_size = 20
        start_page = (start - 1) // page_size + 1
        end_page = (end - 1) // page_size + 1
        
        for page in range(start_page, end_page + 1):
            params = {
                'AllField': keyword,
                'pageSize': page_size,
                'startPage': page - 1
            }
            
            try:
                response = self.session.get(self.search_url, params=params, timeout=30)
                response.raise_for_status()
                
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 查找文章链接
                article_items = soup.find_all('div', class_='item__body')
                
                for item in article_items:
                    title_tag = item.find('a', class_='publication_title')
                    if title_tag:
                        title = title_tag.get_text(strip=True)
                        doi_link = title_tag.get('href', '')
                        
                        if doi_link:
                            full_url = urljoin(self.base_url, doi_link)
                            articles.append({
                                'title': title,
                                'url': full_url,
                                'doi': doi_link
                            })
                
                print(f"第 {page} 页: 找到 {len(article_items)} 篇文章")
                time.sleep(2)  # 避免请求过快
                
            except Exception as e:
                print(f"搜索第 {page} 页时出错: {str(e)}")
                continue
        
        # 根据start和end截取
        articles = articles[start-1:end]
        print(f"共找到 {len(articles)} 篇文章（第{start}-{end}篇）")
        return articles
    
    def download_article(self, article, format_type='pdf'):
        """下载单篇文章"""
        title = article['title']
        url = article['url']
        
        # 清理文件名
        safe_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
        safe_title = safe_title[:100]  # 限制文件名长度
        
        print(f"\n正在下载: {title}")
        
        try:
            # 访问文章页面
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            
            if format_type.lower() == 'pdf':
                return self._download_pdf(response, safe_title, url)
            elif format_type.lower() == 'html':
                return self._download_html(response, safe_title)
            elif format_type.lower() == 'xml':
                return self._download_xml(response, safe_title, url)
            else:
                print(f"不支持的格式: {format_type}")
                return False
                
        except Exception as e:
            print(f"下载失败: {str(e)}")
            return False
    
    def _download_pdf(self, response, filename, url):
        """下载PDF格式"""
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找PDF下载链接
        pdf_link = soup.find('a', class_='pdf-download')
        if not pdf_link:
            pdf_link = soup.find('a', href=lambda x: x and 'pdf' in x.lower())
        
        if pdf_link:
            pdf_url = urljoin(self.base_url, pdf_link.get('href'))
            
            try:
                pdf_response = self.session.get(pdf_url, timeout=60)
                pdf_response.raise_for_status()
                
                filepath = os.path.join(self.output_dir, 'pdf', f"{filename}.pdf")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                with open(filepath, 'wb') as f:
                    f.write(pdf_response.content)
                
                print(f"✓ PDF已保存: {filepath}")
                return True
            except Exception as e:
                print(f"✗ PDF下载失败: {str(e)}")
                return False
        else:
            print("✗ 未找到PDF下载链接")
            return False
    
    def _download_html(self, response, filename):
        """下载HTML格式"""
        try:
            filepath = os.path.join(self.output_dir, 'html', f"{filename}.html")
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"✓ HTML已保存: {filepath}")
            return True
        except Exception as e:
            print(f"✗ HTML保存失败: {str(e)}")
            return False
    
    def _download_xml(self, response, filename, url):
        """下载XML格式"""
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # 查找XML下载链接
        xml_link = soup.find('a', href=lambda x: x and 'xml' in x.lower())
        
        if xml_link:
            xml_url = urljoin(self.base_url, xml_link.get('href'))
            
            try:
                xml_response = self.session.get(xml_url, timeout=30)
                xml_response.raise_for_status()
                
                filepath = os.path.join(self.output_dir, 'xml', f"{filename}.xml")
                os.makedirs(os.path.dirname(filepath), exist_ok=True)
                
                with open(filepath, 'wb') as f:
                    f.write(xml_response.content)
                
                print(f"✓ XML已保存: {filepath}")
                return True
            except Exception as e:
                print(f"✗ XML下载失败: {str(e)}")
                return False
        else:
            print("✗ 未找到XML下载链接")
            return False
    
    def batch_download(self, keyword, start=1, end=10, format_type='pdf'):
        """批量下载文献"""
        print(f"\n{'='*60}")
        print(f"开始批量下载")
        print(f"关键词: {keyword}")
        print(f"范围: 第{start}-{end}篇")
        print(f"格式: {format_type.upper()}")
        print(f"{'='*60}\n")
        
        # 搜索文章
        articles = self.search_articles(keyword, start, end)
        
        if not articles:
            print("未找到任何文章")
            return
        
        # 下载文章
        success_count = 0
        for i, article in enumerate(articles, 1):
            print(f"\n[{i}/{len(articles)}]")
            if self.download_article(article, format_type):
                success_count += 1
            time.sleep(3)  # 避免请求过快
        
        print(f"\n{'='*60}")
        print(f"下载完成: {success_count}/{len(articles)} 篇成功")
        print(f"{'='*60}")


def main():
    print("Wiley文献批量下载工具")
    print("="*60)
    
    # 获取用户输入
    keyword = input("请输入搜索关键词: ").strip()
    
    format_type = input("请选择下载格式 (pdf/html/xml) [默认: pdf]: ").strip().lower()
    if not format_type:
        format_type = 'pdf'
    
    start = input("请输入起始位置 [默认: 1]: ").strip()
    start = int(start) if start else 1
    
    end = input("请输入结束位置 [默认: 10]: ").strip()
    end = int(end) if end else 10
    
    output_dir = input("请输入保存目录 [默认: downloads]: ").strip()
    if not output_dir:
        output_dir = 'downloads'
    
    # 创建下载器并开始下载
    downloader = WileyDownloader(output_dir)
    downloader.batch_download(keyword, start, end, format_type)


if __name__ == "__main__":
    main()
