"""
Legacy manual component tests.

Skipped by default to avoid network/credential dependencies during CI.
Run explicitly if you want to exercise live integrations.
"""
import pytest

pytest.skip("Legacy live component probes; skip during automated test runs", allow_module_level=True)


def test_notion_connection():
    """Test Notion API connection"""
    console.print("\n[bold cyan]Testing Notion Connection...[/bold cyan]")
    
    try:
        from src.notion_client import NotionTranscriptClient
        
        notion = NotionTranscriptClient()
        console.print("✅ Notion client initialized")
        
        # Try to fetch transcripts
        transcripts = notion.fetch_new_transcripts(minutes_ago=10080)  # 1 week
        console.print(f"✅ Found {len(transcripts)} transcripts in last week")
        
        if transcripts:
            console.print("\n[bold]Sample transcript:[/bold]")
            sample = transcripts[0]
            title = notion.get_page_title(sample)
            console.print(f"  Title: {title}")
            console.print(f"  ID: {sample['id'][:16]}...")
            
        return True
        
    except Exception as e:
        console.print(f"[bold red]❌ Notion test failed: {str(e)}[/bold red]")
        return False


def test_gemini_connection():
    """Test Gemini API connection"""
    console.print("\n[bold cyan]Testing Gemini API...[/bold cyan]")
    
    try:
        from llama_index.llms.gemini import Gemini
        
        gemini_key = os.getenv("GEMINI_API_KEY")
        llm = Gemini(
            model="models/gemini-2.0-flash-exp",
            api_key=gemini_key,
            temperature=0.7
        )
        
        # Test completion
        response = llm.complete("Say 'Hello from Gemini!' in exactly those words.")
        console.print(f"✅ Gemini response: {response.text[:100]}")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]❌ Gemini test failed: {str(e)}[/bold red]")
        return False


def test_pinecone_connection():
    """Test Pinecone connection"""
    console.print("\n[bold cyan]Testing Pinecone Connection...[/bold cyan]")
    
    try:
        from src.pinecone_client import PineconeClient
        
        pinecone = PineconeClient()
        console.print("✅ Pinecone client initialized")
        
        # Get index info
        info = pinecone.get_index_info()
        
        table = Table(title="Pinecone Index Info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")
        
        for key, value in info.items():
            if key != 'host':
                table.add_row(str(key), str(value))
        
        console.print(table)
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]❌ Pinecone test failed: {str(e)}[/bold red]")
        return False


def test_full_processor():
    """Test the full LLM processor"""
    console.print("\n[bold cyan]Testing Full Processor...[/bold cyan]")
    
    try:
        from src.llm_processor import TranscriptProcessor
        
        processor = TranscriptProcessor()
        console.print("✅ Processor initialized")
        
        # Test with sample text
        sample_text = """
        Today I had an interesting conversation about productivity and time management.
        We discussed the Pomodoro technique and how it can help with focus.
        I also learned about the importance of taking breaks and avoiding burnout.
        The key insight was that sustainable productivity requires balance.
        """
        
        console.print("\n[bold]Testing theme extraction...[/bold]")
        themes = processor.extract_themes(sample_text)
        console.print(f"✅ Extracted themes: {themes}")
        
        # Get index stats
        stats = processor.get_index_stats()
        console.print(f"\n✅ Index contains {stats.get('total_vectors', 0)} vectors")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]❌ Processor test failed: {str(e)}[/bold red]")
        return False


def test_visualizer():
    """Test the visualizer"""
    console.print("\n[bold cyan]Testing Visualizer...[/bold cyan]")
    
    try:
        from src.visualizer import MindMapGenerator
        
        viz = MindMapGenerator()
        console.print("✅ Visualizer initialized")
        
        # Add test nodes
        viz.add_transcript_node(
            "test-1",
            "Test Transcript 1",
            ["productivity", "focus"],
            centrality=0.8
        )
        viz.add_transcript_node(
            "test-2",
            "Test Transcript 2",
            ["productivity", "wellness"],
            centrality=0.6
        )
        viz.add_connection("test-1", "test-2", "relates_to", 1.0)
        
        # Generate stats
        stats = viz.generate_statistics()
        console.print(f"✅ Graph has {stats['total_nodes']} nodes and {stats['total_edges']} edges")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]❌ Visualizer test failed: {str(e)}[/bold red]")
        return False


def run_all_tests():
    """Run all component tests"""
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]🧪 PLAUDBLENDER COMPONENT TESTS[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]")
    
    tests = [
        ("Notion Connection", test_notion_connection),
        ("Gemini API", test_gemini_connection),
        ("Pinecone Connection", test_pinecone_connection),
        ("Full Processor", test_full_processor),
        ("Visualizer", test_visualizer),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            console.print(f"[bold red]Unexpected error in {test_name}: {e}[/bold red]")
            results.append((test_name, False))
    
    # Summary
    console.print("\n[bold magenta]" + "=" * 60 + "[/bold magenta]")
    console.print("[bold magenta]📊 TEST SUMMARY[/bold magenta]")
    console.print("[bold magenta]" + "=" * 60 + "[/bold magenta]\n")
    
    summary_table = Table(show_header=True, header_style="bold magenta")
    summary_table.add_column("Test", style="cyan", width=30)
    summary_table.add_column("Status", justify="center", width=15)
    
    passed = 0
    for test_name, result in results:
        status = "[bold green]✅ PASSED[/bold green]" if result else "[bold red]❌ FAILED[/bold red]"
        summary_table.add_row(test_name, status)
        if result:
            passed += 1
    
    console.print(summary_table)
    
    console.print(f"\n[bold]Results: {passed}/{len(results)} tests passed[/bold]")
    
    if passed == len(results):
        console.print("\n[bold green]🎉 All tests passed! System is ready.[/bold green]\n")
        return True
    else:
        console.print("\n[bold yellow]⚠️  Some tests failed. Check configuration.[/bold yellow]\n")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test PlaudBlender components")
    parser.add_argument('--notion', action='store_true', help='Test only Notion')
    parser.add_argument('--gemini', action='store_true', help='Test only Gemini')
    parser.add_argument('--pinecone', action='store_true', help='Test only Pinecone')
    parser.add_argument('--processor', action='store_true', help='Test only processor')
    parser.add_argument('--visualizer', action='store_true', help='Test only visualizer')
    
    args = parser.parse_args()
    
    # Run specific test if requested
    if args.notion:
        test_notion_connection()
    elif args.gemini:
        test_gemini_connection()
    elif args.pinecone:
        test_pinecone_connection()
    elif args.processor:
        test_full_processor()
    elif args.visualizer:
        test_visualizer()
    else:
        # Run all tests
        success = run_all_tests()
        sys.exit(0 if success else 1)
