#!/usr/bin/env python3
"""
Comprehensive Chunk Analysis Script for TDS Forum Data

This script analyzes the generated chunks.json file to evaluate the effectiveness
of the chunking strategy and provide insights for optimization.
"""

import json
import re
import statistics
import tiktoken
from collections import Counter, defaultdict
from typing import List, Dict, Any, Tuple
from datetime import datetime


class ChunkAnalyzer:
    """Analyzes chunked forum data to         report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Chunks File: {self.chunks_path}")
    report.append("")luate chunking effectiveness."""

    def __init__(self, chunks_path: str, original_data_path: str = None):
        """Initialize with paths to chunked data and optionally original data."""
        self.chunks_path = chunks_path
        self.original_data_path = original_data_path
        self.chunks = self._load_chunks()
        self.original_data = self._load_original_data() if original_data_path else None
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.analysis_results = {}

    def _load_chunks(self) -> List[Dict[str, Any]]:
        """Load and return the chunks data."""
        try:
            with open(self.chunks_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Error: File {self.chunks_path} not found")
            return []
        except json.JSONDecodeError as e:
            print(f"Error parsing chunks JSON: {e}")
            return []

    def _load_original_data(self) -> List[Dict[str, Any]]:
        """Load and return the original data for comparison."""
        try:
            with open(self.original_data_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: Could not load original data: {e}")
            return []

    def basic_chunk_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics about the chunks."""
        if not self.chunks:
            return {}

        # Token and character analysis
        token_counts = [
            len(self.tokenizer.encode(chunk["text"])) for chunk in self.chunks
        ]
        char_counts = [len(chunk["text"]) for chunk in self.chunks]
        word_counts = [len(chunk["text"].split()) for chunk in self.chunks]

        # Topic distribution
        topic_distribution = Counter(chunk["topic_id"] for chunk in self.chunks)
        chunks_per_topic = list(topic_distribution.values())

        # Chunk ID analysis to see multi-chunk topics
        multi_chunk_topics = sum(
            1 for count in topic_distribution.values() if count > 1
        )
        single_chunk_topics = sum(
            1 for count in topic_distribution.values() if count == 1
        )

        stats = {
            "total_chunks": len(self.chunks),
            "unique_topics": len(topic_distribution),
            "multi_chunk_topics": multi_chunk_topics,
            "single_chunk_topics": single_chunk_topics,
            "token_stats": {
                "mean": statistics.mean(token_counts),
                "median": statistics.median(token_counts),
                "std": statistics.stdev(token_counts) if len(token_counts) > 1 else 0,
                "min": min(token_counts),
                "max": max(token_counts),
                "percentiles": {
                    "25th": (
                        statistics.quantiles(token_counts, n=4)[0]
                        if token_counts
                        else 0
                    ),
                    "75th": (
                        statistics.quantiles(token_counts, n=4)[2]
                        if token_counts
                        else 0
                    ),
                    "90th": (
                        statistics.quantiles(token_counts, n=10)[8]
                        if token_counts
                        else 0
                    ),
                    "95th": (
                        statistics.quantiles(token_counts, n=20)[18]
                        if token_counts
                        else 0
                    ),
                },
            },
            "character_stats": {
                "mean": statistics.mean(char_counts),
                "median": statistics.median(char_counts),
                "min": min(char_counts),
                "max": max(char_counts),
            },
            "word_stats": {
                "mean": statistics.mean(word_counts),
                "median": statistics.median(word_counts),
                "min": min(word_counts),
                "max": max(word_counts),
            },
            "chunks_per_topic_stats": {
                "mean": statistics.mean(chunks_per_topic),
                "median": statistics.median(chunks_per_topic),
                "max": max(chunks_per_topic),
                "distribution": dict(Counter(chunks_per_topic)),
            },
        }

        self.analysis_results["basic_stats"] = stats
        return stats

    def chunking_quality_assessment(self) -> Dict[str, Any]:
        """Assess the quality of the chunking strategy."""
        token_counts = [
            len(self.tokenizer.encode(chunk["text"])) for chunk in self.chunks
        ]

        # Token limit compliance
        MAX_TOKENS = 500  # From the chunking script
        over_limit = sum(1 for count in token_counts if count > MAX_TOKENS)
        near_limit = sum(1 for count in token_counts if 450 <= count <= MAX_TOKENS)
        under_utilized = sum(1 for count in token_counts if count < 100)

        # Content preservation analysis
        content_issues = self._analyze_content_preservation()

        # Overlap analysis
        overlap_analysis = self._analyze_overlap()

        quality_metrics = {
            "token_limit_compliance": {
                "total_chunks": len(token_counts),
                "over_limit": over_limit,
                "over_limit_percentage": (over_limit / len(token_counts)) * 100,
                "near_limit": near_limit,
                "near_limit_percentage": (near_limit / len(token_counts)) * 100,
                "under_utilized": under_utilized,
                "under_utilized_percentage": (under_utilized / len(token_counts)) * 100,
            },
            "content_preservation": content_issues,
            "overlap_analysis": overlap_analysis,
            "size_consistency": {
                "coefficient_of_variation": statistics.stdev(token_counts)
                / statistics.mean(token_counts),
                "size_range": max(token_counts) - min(token_counts),
            },
        }

        self.analysis_results["quality_metrics"] = quality_metrics
        return quality_metrics

    def _analyze_content_preservation(self) -> Dict[str, Any]:
        """Analyze how well the chunking preserves content integrity."""
        issues = {
            "broken_conversations": 0,
            "incomplete_code_blocks": 0,
            "broken_urls": 0,
            "mid_sentence_breaks": 0,
            "lost_context": 0,
        }

        for chunk in self.chunks:
            text = chunk["text"]

            # Check for mid-sentence breaks (very crude heuristic)
            if not text.strip().endswith((".", "!", "?", ":", "\n")):
                if not text.strip().endswith(("...", "==", ")", "]", "}")):
                    issues["mid_sentence_breaks"] += 1

            # Check for incomplete code blocks
            code_block_starts = text.count("```")
            if code_block_starts % 2 != 0:
                issues["incomplete_code_blocks"] += 1

            # Check for broken URLs (very basic)
            if "http" in text and not re.search(r"https?://[^\s]+", text):
                issues["broken_urls"] += 1

            # Check for conversation breaks (username appears but no following text)
            usernames = re.findall(r"^[a-zA-Z0-9._@]+:$", text, re.MULTILINE)
            if usernames and text.strip().endswith(":"):
                issues["broken_conversations"] += 1

        return issues

    def _analyze_overlap(self) -> Dict[str, Any]:
        """Analyze overlap between chunks from the same topic."""
        overlap_analysis = {
            "topics_with_overlap": 0,
            "average_overlap_length": 0,
            "overlap_examples": [],
        }

        # Group chunks by topic
        topic_chunks = defaultdict(list)
        for chunk in self.chunks:
            topic_chunks[chunk["topic_id"]].append(chunk)

        total_overlaps = []
        topics_with_overlap = 0

        for topic_id, chunks in topic_chunks.items():
            if len(chunks) > 1:
                # Check consecutive chunks for overlap
                for i in range(len(chunks) - 1):
                    chunk1_text = chunks[i]["text"]
                    chunk2_text = chunks[i + 1]["text"]

                    # Look for overlap at the end of chunk1 and beginning of chunk2
                    overlap_length = self._find_overlap_length(chunk1_text, chunk2_text)
                    if overlap_length > 0:
                        total_overlaps.append(overlap_length)
                        if len(overlap_analysis["overlap_examples"]) < 3:
                            overlap_analysis["overlap_examples"].append(
                                {
                                    "topic_id": topic_id,
                                    "chunk_ids": [
                                        chunks[i]["chunk_id"],
                                        chunks[i + 1]["chunk_id"],
                                    ],
                                    "overlap_length": overlap_length,
                                }
                            )

        if total_overlaps:
            topics_with_overlap = len(
                set(
                    example["topic_id"]
                    for example in overlap_analysis["overlap_examples"]
                )
            )
            overlap_analysis["topics_with_overlap"] = topics_with_overlap
            overlap_analysis["average_overlap_length"] = statistics.mean(total_overlaps)
            overlap_analysis["total_overlaps_found"] = len(total_overlaps)

        return overlap_analysis

    def _find_overlap_length(self, text1: str, text2: str) -> int:
        """Find character overlap between end of text1 and beginning of text2."""
        max_overlap = min(len(text1), len(text2), 500)  # Limit search

        for i in range(max_overlap, 10, -1):  # Minimum 10 char overlap
            if text1[-i:] == text2[:i]:
                return i
        return 0

    def content_type_analysis(self) -> Dict[str, Any]:
        """Analyze different types of content in chunks."""
        content_types = {
            "code_blocks": 0,
            "urls": 0,
            "mentions": 0,
            "questions": 0,
            "answers": 0,
            "assignments": 0,
            "error_messages": 0,
        }

        for chunk in self.chunks:
            text = chunk["text"]

            # Code blocks
            if "```" in text or "def " in text or "import " in text:
                content_types["code_blocks"] += 1

            # URLs
            if re.search(r"https?://[^\s]+", text):
                content_types["urls"] += 1

            # Mentions
            if "@" in text:
                content_types["mentions"] += 1

            # Questions (heuristic)
            if "?" in text or any(
                q in text.lower() for q in ["how to", "what is", "why", "help"]
            ):
                content_types["questions"] += 1

            # Assignment related
            if any(
                term in text.lower()
                for term in ["assignment", "ga1", "ga2", "ga3", "deadline"]
            ):
                content_types["assignments"] += 1

            # Error messages
            if any(
                term in text.lower()
                for term in ["error", "exception", "failed", "not working"]
            ):
                content_types["error_messages"] += 1

        self.analysis_results["content_types"] = content_types
        return content_types

    def compare_with_original(self) -> Dict[str, Any]:
        """Compare chunks with original data if available."""
        if not self.original_data:
            return {"comparison_available": False}

        original_stats = {
            "total_topics": len(self.original_data),
            "total_posts": sum(len(topic["posts"]) for topic in self.original_data),
            "avg_posts_per_topic": 0,
        }

        if original_stats["total_topics"] > 0:
            original_stats["avg_posts_per_topic"] = (
                original_stats["total_posts"] / original_stats["total_topics"]
            )

        # Calculate original content lengths
        original_post_lengths = []
        for topic in self.original_data:
            for post in topic["posts"]:
                original_post_lengths.append(len(post["text"]))

        original_stats["post_length_stats"] = {
            "mean": (
                statistics.mean(original_post_lengths) if original_post_lengths else 0
            ),
            "total_characters": sum(original_post_lengths),
        }

        # Compare with chunks
        chunk_stats = self.analysis_results.get("basic_stats", {})
        total_chunk_chars = sum(len(chunk["text"]) for chunk in self.chunks)

        comparison = {
            "comparison_available": True,
            "original_stats": original_stats,
            "compression_ratio": (
                total_chunk_chars
                / original_stats["post_length_stats"]["total_characters"]
                if original_stats["post_length_stats"]["total_characters"] > 0
                else 0
            ),
            "topics_preserved": (
                chunk_stats.get("unique_topics", 0) / original_stats["total_topics"]
                if original_stats["total_topics"] > 0
                else 0
            ),
            "chunking_factor": (
                chunk_stats.get("total_chunks", 0) / original_stats["total_topics"]
                if original_stats["total_topics"] > 0
                else 0
            ),
        }

        self.analysis_results["comparison"] = comparison
        return comparison

    def generate_recommendations(self) -> Dict[str, Any]:
        """Generate recommendations for improving the chunking strategy."""
        quality_metrics = self.analysis_results.get("quality_metrics", {})
        basic_stats = self.analysis_results.get("basic_stats", {})

        recommendations = {
            "overall_assessment": "good",  # Will be updated based on analysis
            "specific_recommendations": [],
            "parameter_suggestions": {},
            "alternative_strategies": [],
        }

        # Assess token limit compliance
        token_compliance = quality_metrics.get("token_limit_compliance", {})
        over_limit_pct = token_compliance.get("over_limit_percentage", 0)
        under_utilized_pct = token_compliance.get("under_utilized_percentage", 0)

        if over_limit_pct > 5:
            recommendations["overall_assessment"] = "needs_improvement"
            recommendations["specific_recommendations"].append(
                f"Reduce MAX_TOKENS or improve sentence splitting - {over_limit_pct:.1f}% of chunks exceed token limit"
            )
            recommendations["parameter_suggestions"]["max_tokens"] = 450

        if under_utilized_pct > 30:
            recommendations["specific_recommendations"].append(
                f"Increase chunk size utilization - {under_utilized_pct:.1f}% of chunks are under-utilized"
            )
            recommendations["parameter_suggestions"]["min_tokens"] = 150

        # Assess content preservation
        content_issues = quality_metrics.get("content_preservation", {})
        if content_issues.get("mid_sentence_breaks", 0) > 5:
            recommendations["specific_recommendations"].append(
                "Improve sentence boundary detection to avoid mid-sentence breaks"
            )

        if content_issues.get("incomplete_code_blocks", 0) > 0:
            recommendations["specific_recommendations"].append(
                "Implement code block aware chunking to preserve code integrity"
            )

        # Size consistency assessment
        size_consistency = quality_metrics.get("size_consistency", {})
        cv = size_consistency.get("coefficient_of_variation", 0)
        if cv > 0.5:
            recommendations["specific_recommendations"].append(
                "High size variability detected - consider more consistent chunking approach"
            )

        # Alternative strategies
        if over_limit_pct > 10 or under_utilized_pct > 40:
            recommendations["alternative_strategies"].append(
                {
                    "name": "adaptive_chunking",
                    "description": "Dynamically adjust chunk size based on content type",
                    "benefits": "Better size utilization and content preservation",
                }
            )

        if content_issues.get("broken_conversations", 0) > 5:
            recommendations["alternative_strategies"].append(
                {
                    "name": "conversation_aware_chunking",
                    "description": "Chunk based on conversation boundaries rather than token limits",
                    "benefits": "Preserves conversational context and Q&A relationships",
                }
            )

        # Overlap recommendations
        overlap_analysis = quality_metrics.get("overlap_analysis", {})
        avg_overlap = overlap_analysis.get("average_overlap_length", 0)
        if avg_overlap < 50:
            recommendations["parameter_suggestions"]["overlap_tokens"] = 50
            recommendations["specific_recommendations"].append(
                "Increase overlap to improve context continuity between chunks"
            )
        elif avg_overlap > 200:
            recommendations["parameter_suggestions"]["overlap_tokens"] = 100
            recommendations["specific_recommendations"].append(
                "Reduce overlap to minimize redundancy"
            )

        self.analysis_results["recommendations"] = recommendations
        return recommendations

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        if not self.analysis_results:
            self.run_full_analysis()

        report = []
        report.append("=" * 80)
        report.append("COMPREHENSIVE CHUNK ANALYSIS REPORT")
        report.append("=" * 80)
        report.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Chunks File: {self.chunks_path}")
        report.append()

        # Basic Statistics
        basic_stats = self.analysis_results.get("basic_stats", {})
        report.append("📊 CHUNK STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Chunks: {basic_stats.get('total_chunks', 0):,}")
        report.append(f"Unique Topics: {basic_stats.get('unique_topics', 0):,}")
        report.append(
            f"Multi-chunk Topics: {basic_stats.get('multi_chunk_topics', 0):,}"
        )
        report.append(
            f"Single-chunk Topics: {basic_stats.get('single_chunk_topics', 0):,}"
        )
        report.append()

        # Token Analysis
        token_stats = basic_stats.get("token_stats", {})
        report.append("🔢 TOKEN ANALYSIS")
        report.append("-" * 40)
        report.append(f"Average Tokens: {token_stats.get('mean', 0):.1f}")
        report.append(f"Median Tokens: {token_stats.get('median', 0):.1f}")
        report.append(
            f"Token Range: {token_stats.get('min', 0)} - {token_stats.get('max', 0)}"
        )
        report.append(f"Standard Deviation: {token_stats.get('std', 0):.1f}")

        percentiles = token_stats.get("percentiles", {})
        report.append(f"25th Percentile: {percentiles.get('25th', 0):.0f} tokens")
        report.append(f"75th Percentile: {percentiles.get('75th', 0):.0f} tokens")
        report.append(f"90th Percentile: {percentiles.get('90th', 0):.0f} tokens")
        report.append(f"95th Percentile: {percentiles.get('95th', 0):.0f} tokens")
        report.append()

        # Quality Assessment
        quality_metrics = self.analysis_results.get("quality_metrics", {})
        token_compliance = quality_metrics.get("token_limit_compliance", {})
        report.append("✅ QUALITY ASSESSMENT")
        report.append("-" * 40)
        report.append(
            f"Chunks Over Token Limit: {token_compliance.get('over_limit', 0)} ({token_compliance.get('over_limit_percentage', 0):.1f}%)"
        )
        report.append(
            f"Chunks Near Token Limit: {token_compliance.get('near_limit', 0)} ({token_compliance.get('near_limit_percentage', 0):.1f}%)"
        )
        report.append(
            f"Under-utilized Chunks: {token_compliance.get('under_utilized', 0)} ({token_compliance.get('under_utilized_percentage', 0):.1f}%)"
        )

        # Content Preservation Issues
        content_issues = quality_metrics.get("content_preservation", {})
        if any(content_issues.values()):
            report.append()
            report.append("⚠️  CONTENT PRESERVATION ISSUES")
            report.append("-" * 40)
            for issue_type, count in content_issues.items():
                if count > 0:
                    report.append(f"{issue_type.replace('_', ' ').title()}: {count}")

        # Overlap Analysis
        overlap_analysis = quality_metrics.get("overlap_analysis", {})
        if overlap_analysis.get("total_overlaps_found", 0) > 0:
            report.append()
            report.append("🔗 OVERLAP ANALYSIS")
            report.append("-" * 40)
            report.append(
                f"Topics with Overlap: {overlap_analysis.get('topics_with_overlap', 0)}"
            )
            report.append(
                f"Average Overlap Length: {overlap_analysis.get('average_overlap_length', 0):.1f} characters"
            )
            report.append(
                f"Total Overlaps Found: {overlap_analysis.get('total_overlaps_found', 0)}"
            )

        # Content Types
        content_types = self.analysis_results.get("content_types", {})
        if content_types:
            report.append()
            report.append("📝 CONTENT TYPE DISTRIBUTION")
            report.append("-" * 40)
            for content_type, count in content_types.items():
                percentage = (count / basic_stats.get("total_chunks", 1)) * 100
                report.append(
                    f"{content_type.replace('_', ' ').title()}: {count} ({percentage:.1f}%)"
                )

        # Comparison with Original
        comparison = self.analysis_results.get("comparison", {})
        if comparison.get("comparison_available", False):
            report.append()
            report.append("📋 COMPARISON WITH ORIGINAL DATA")
            report.append("-" * 40)
            report.append(
                f"Topics Preserved: {comparison.get('topics_preserved', 0):.1%}"
            )
            report.append(
                f"Chunking Factor: {comparison.get('chunking_factor', 0):.1f} chunks per topic"
            )
            report.append(
                f"Compression Ratio: {comparison.get('compression_ratio', 0):.1%}"
            )

        # Recommendations
        recommendations = self.analysis_results.get("recommendations", {})
        report.append()
        report.append("🎯 RECOMMENDATIONS")
        report.append("-" * 40)

        assessment = recommendations.get("overall_assessment", "unknown")
        report.append(f"Overall Assessment: {assessment.upper()}")
        report.append()

        specific_recs = recommendations.get("specific_recommendations", [])
        if specific_recs:
            report.append("Specific Recommendations:")
            for i, rec in enumerate(specific_recs, 1):
                report.append(f"{i}. {rec}")
            report.append()

        param_suggestions = recommendations.get("parameter_suggestions", {})
        if param_suggestions:
            report.append("Parameter Suggestions:")
            for param, value in param_suggestions.items():
                report.append(f"• {param}: {value}")
            report.append()

        alt_strategies = recommendations.get("alternative_strategies", [])
        if alt_strategies:
            report.append("Alternative Strategies:")
            for strategy in alt_strategies:
                report.append(f"• {strategy['name']}: {strategy['description']}")
                report.append(f"  Benefits: {strategy['benefits']}")
            report.append()

        report.append("=" * 80)

        return "\n".join(report)

    def run_full_analysis(self):
        """Run complete analysis pipeline."""
        print("🔍 Analyzing chunks...")
        self.basic_chunk_statistics()
        self.chunking_quality_assessment()
        self.content_type_analysis()
        self.compare_with_original()
        self.generate_recommendations()
        print("✅ Analysis complete!")

    def save_results(self, output_path: str):
        """Save analysis results to JSON file."""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)
        print(f"📁 Results saved to {output_path}")


def main():
    """Main execution function."""
    chunks_path = "chunks.json"
    original_data_path = "tds_cleaned_data.json"

    print("🚀 Starting Comprehensive Chunk Analysis")
    print("=" * 50)

    # Initialize analyzer
    analyzer = ChunkAnalyzer(chunks_path, original_data_path)

    if not analyzer.chunks:
        print("❌ No chunks loaded. Please check the file path.")
        return

    # Run analysis
    analyzer.run_full_analysis()

    # Generate and display report
    report = analyzer.generate_comprehensive_report()
    print(report)

    # Save results
    analyzer.save_results("chunk_analysis_results.json")

    # Save report to file
    with open("chunk_analysis_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print("📄 Report saved to chunk_analysis_report.txt")

    print(
        "\n🎉 Chunk analysis complete! Check the generated files for detailed results."
    )


if __name__ == "__main__":
    main()
