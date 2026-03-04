#!/usr/bin/env python3
"""Quick test script to debug Gemini API responses with detailed logging."""

import json
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)

logger = logging.getLogger(__name__)

def test_gemini_structured_output():
    """Test Gemini structured output with a simple example."""
    
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.error("No API key found! Set GEMINI_API_KEY or GOOGLE_API_KEY")
        return False
    
    logger.info("Testing Gemini structured output...")
    
    try:
        from google import genai
        from pydantic import BaseModel
        from typing import Literal
        
        # Simple test schema
        class TestResponse(BaseModel):
            action: Literal["YES", "NO"]
            reason: str
            confidence: float
        
        client = genai.Client(api_key=api_key)
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents="Should I invest in stocks? Give a simple answer.",
            config={
                "response_mime_type": "application/json",
                "response_json_schema": TestResponse.model_json_schema(),
                "temperature": 0.2,
            },
        )
        
        logger.info(f"Response type: {type(response)}")
        logger.info(f"Response has 'text' attr: {hasattr(response, 'text')}")
        
        if hasattr(response, 'text'):
            raw_text = response.text
            logger.info(f"Response length: {len(raw_text)} chars")
            logger.debug(f"Raw text: {raw_text}")
            
            data = json.loads(raw_text)
            validated = TestResponse.model_validate(data)
            
            logger.info(f"✅ SUCCESS!")
            logger.info(f"  Action: {validated.action}")
            logger.info(f"  Reason: {validated.reason}")
            logger.info(f"  Confidence: {validated.confidence}")
            return True
        else:
            logger.error("Response does not have 'text' attribute")
            logger.debug(f"Response dir: {dir(response)}")
            return False
            
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Run: pip install google-genai")
        return False
    except Exception as e:
        logger.error(f"Test failed: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("GEMINI API STRUCTURED OUTPUT TEST")
    logger.info("="*60)
    
    success = test_gemini_structured_output()
    
    logger.info("="*60)
    if success:
        logger.info("✅ Test PASSED - Gemini API is working correctly")
        sys.exit(0)
    else:
        logger.error("❌ Test FAILED - Check errors above")
        sys.exit(1)
