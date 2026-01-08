from openai import OpenAI
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
from docling.pipeline.standard_pdf_pipeline import StandardPdfPipeline
from docling.datamodel.settings import settings
from jinja2 import Environment, FileSystemLoader
from langchain_openai import ChatOpenAI
from datetime import datetime
from tqdm import tqdm
import uuid
import spacy
import re
import random
import numpy as np
import json
import logging
import sys
import subprocess
import click
import os


class DoclingPdfReader:
    def __init__(self, page_batch_size:int=4, use_gpu:bool=True):
        pipeline_options = PdfPipelineOptions(
            ocr_options=RapidOcrOptions(
                backend='torch'
            )
        )
        if use_gpu:
            pipeline_options.accelerator_options = AcceleratorOptions(
                device=AcceleratorDevice.CUDA
            )

        settings.perf.page_batch_size = page_batch_size

        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=StandardPdfPipeline,
                    pipeline_options=pipeline_options
                )
            }
        )

        self.converter.initialize_pipeline(InputFormat.PDF)

    def extract(self, path:str):
        docs = self.converter.convert(path)
        return docs.document.export_to_markdown(image_placeholder="")
    
class WordAwareLegalSplitter:
    """
    Word-based splitter for legal documents
    Respects sentence boundaries using spaCy
    """
    
    def __init__(
        self,
        nlp,
        chunk_size_words: int = 150,
        chunk_overlap_words: int = 30,
        max_chunk_size_words: int = 300
    ):
        self.chunk_size = chunk_size_words
        self.overlap = chunk_overlap_words
        self.max_size = max_chunk_size_words
        self.nlp = nlp
    
    def preprocess(self, text: str) -> str:
        """Add structural breaks"""
        text = re.sub(
            r'(\n)([-=\s]*(?:PASAL|Pasal|BAB|Bab)\s+(?:[IVXLCDM]+|\d+)(?:\s*[-:]\s*[^\n]*)?[-=\s]*)(\n)',
            r'\1\1\2\3\3',
            text
        )
        text = re.sub(
            r'(\n)((?:BAGIAN|Bagian|PARAGRAF|Paragraf|AYAT|Ayat|KETENTUAN)\s+[^\n]+)(\n)',
            r'\1\1\2\3\3',
            text
        )
        text = re.sub(r'-{5,}', '---', text)
        text = re.sub(r'={5,}', '===', text)
        text = re.sub(r'(\n\n)(\d+\.\s+[A-Z])', r'\1\1\2', text)
        text = re.sub(r'(\n)([A-Z][a-zA-Z\s&]+)(\n)', r'\1\1\2\3', text)
        text = re.sub(r'(\n)(•)', r'\1\1\2', text)
        text = re.sub(r'(\n)(Figure\s+\d+)', r'\1\1\2', text)
        text = re.sub(r'\n{4,}', '\n\n\n', text)
        return text
    
    def is_header(self, text: str) -> bool:
        """Detect if a paragraph is a legal header"""
        text = text.strip()
        
        # Remove decorations
        cleaned = re.sub(r'^[-=\s]+', '', text)
        cleaned = re.sub(r'[-=\s]+$', '', cleaned)
        cleaned = cleaned.strip()
        
        # Check if it's a header
        patterns = [
            r'^(?:PASAL|Pasal|BAB|Bab)\s+(?:[IVXLCDM]+|\d+)',
            r'^(?:BAGIAN|Bagian|PARAGRAF|Paragraf|AYAT|Ayat)\s+',
            r'^(?:KETENTUAN\s+(?:UMUM|PENUTUP|PERALIHAN))',
            r'^\d+\.\s+[A-Z]',  # "1. Title"
            r'^##\s+.+',
        ]
        
        for pattern in patterns:
            if re.match(pattern, cleaned):
                # Additional check: headers are typically short
                if len(cleaned) < 200:
                    return True
        
        # Check for decoration-heavy headers
        if text.count('-') > 10 or text.count('=') > 10:
            legal_keywords = ['pasal', 'bab', 'bagian', 'paragraf', 'ayat', 'ketentuan']
            if any(kw in text.lower() for kw in legal_keywords):
                return True
        
        return False
    
    def merge_headers_with_content(self, paragraphs: list[str]) -> list[str]:
        """
        Merge header paragraphs with their following content
        
        CRITICAL: This prevents headers from being orphaned
        """
        merged = []
        i = 0
        
        while i < len(paragraphs):
            current = paragraphs[i]
            
            # Check if current paragraph is a header
            if self.is_header(current):
                # Merge with next paragraph if it exists
                if i + 1 < len(paragraphs):
                    next_para = paragraphs[i + 1]
                    
                    # Merge: "Header\n\nContent"
                    merged_text = f"{current}\n\n{next_para}"
                    merged.append(merged_text)
                    
                    i += 2  # Skip both header and content
                else:
                    # Header is last paragraph (rare but possible)
                    merged.append(current)
                    i += 1
            else:
                # Not a header, keep as-is
                merged.append(current)
                i += 1
        
        return merged
    
    def count_words(self, text: str) -> int:
        """Count words in text"""
        return len(text.split())
    
    def split_text(self, text: str) -> list[str]:
        """Split by words, respecting sentences"""
        # Step 1: Preprocess
        text = self.preprocess(text)
        
        # Step 2: Split by paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        # Step 3: CRITICAL - Merge headers with content
        paragraphs = self.merge_headers_with_content(paragraphs)
        
        chunks = []
        
        # Step 4: Process each paragraph
        for para in paragraphs:
            word_count = self.count_words(para)
            
            if word_count <= self.chunk_size:
                # Paragraph is small enough
                chunks.append(para)
            else:
                # Split by sentences
                doc = self.nlp(para)
                sentences = [sent.text.strip() for sent in doc.sents]
                
                current_chunk = []
                current_word_count = 0
                
                for sent in sentences:
                    sent_words = self.count_words(sent)
                    
                    if current_word_count + sent_words <= self.chunk_size:
                        current_chunk.append(sent)
                        current_word_count += sent_words
                    else:
                        # Save current chunk
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        
                        # Start new chunk
                        current_chunk = [sent]
                        current_word_count = sent_words
                
                # Don't forget last chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
        
        return chunks
    
class TqdmLoggingHandler(logging.Handler):
    """Custom logging handler that uses tqdm.write()"""
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
        except Exception:
            self.handleError(record)

def get_logger(name:str='DataCreator') -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    logger.handlers.clear()
    if not logger.handlers:
        # console_handler = logging.StreamHandler()
        console_handler = TqdmLoggingHandler()
        console_handler.setFormatter(
            logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )
        logger.addHandler(console_handler)
    return logger

LOGGER = get_logger()

def check_spacy_deps():
    def load_en_core():
        try:
            nlp = spacy.load('en_core_web_sm')
            return nlp
        except OSError:
            raise
        except Exception:
            raise

    try:
        LOGGER.info('Checking if spacy deps exists')
        nlp = load_en_core()
        LOGGER.info('required deps exists. Continue...')
        return nlp
    except OSError:
        LOGGER.warning('`en_core_web_sm` deps doesnt found on the system. Installing...')
        try:
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', 'en_core_web_sm'])
            LOGGER.info('Successfully downloaded deps.')
            nlp = load_en_core()
            return nlp
        except Exception as e:
            logging.error('Could not perform deps installation', exc_info=e)
            raise Exception
    except Exception:
        LOGGER.error('Could not load spacy deps, breaking early.')

def embed_chunks(chunks:list[str], base_url_embedder:str, model:str, api_key:str='dummykey'):
    client = OpenAI(base_url=base_url_embedder, api_key=api_key)
    resp = client.embeddings.create(model=model, input=chunks)
    embeddings = [data.embedding[0]['embedding'][0] for data in resp.data]
    embed_dict = [
        {'text': text, 'embedding': emb} for text, emb in zip(chunks, embeddings)
    ]
    return embed_dict


def relevant_context_generator(embeddings:list[dict[str, any]], threshold:float=.8):
    ref_index = random.randint(0, len(embeddings) - 1)
    ref_embedding = embeddings[ref_index]
    contexts = [ref_embedding['text']]
    LOGGER.info('Selecting random chunk as reference chunk to generate question and answer...')

    similar_indices = []
    for i, embedding in enumerate(embeddings):
        product = np.dot(ref_embedding['embedding'], embedding['embedding'])
        norm = np.linalg.norm(ref_embedding['embedding']) * np.linalg.norm(embedding['embedding'])
        sim = product / norm
        if sim >= threshold:
            similar_indices.append(i)

    for i in similar_indices:
        contexts.append(embeddings[i]['text'])

    LOGGER.info(f'Got {len(contexts)} chunks as contexts to generate synth data...')
    return contexts

def get_template(template_path:str, template_dir:str='.'):
    env = Environment(
        loader=FileSystemLoader(template_dir),
        lstrip_blocks=True,
        trim_blocks=True
    )
    return env.get_template(template_path)

def create_questions(
        contexts:list[str], 
        generator_model:str, 
        generator_base_url:str,
        api_key:str,
        generator_temperature:float=.3,
        max_out_tokens:int|None = None,
        num_question_per_context:int=5, 
        question_prompt_path:str='question-prompt.jinja'
        ):
    question_prompt = get_template(question_prompt_path).render(contexts=contexts, num_question_per_context=num_question_per_context)

    try:
        response_content = ChatOpenAI(model=generator_model, base_url=generator_base_url, api_key=api_key, temperature=generator_temperature, max_completion_tokens=max_out_tokens).invoke(question_prompt).content
        questions = json.loads(response_content)
    except json.JSONDecodeError as e:
        LOGGER.error('invalid json structure/values ', e)
        raise
    except Exception as e:
        LOGGER.error('Unexpected exception, error might be coming from OpenRouter ', e)
        raise

    return questions

def get_all_refine_question_template(dir_path:str='.'):
    template_paths = []
    for filename in os.listdir(dir_path):
        parts = filename.split('_')
        if os.path.isfile(os.path.join(dir_path, filename)) and parts[:2] == ['refine', 'question']:
            template_paths.append(filename)

    if not template_paths:
        raise FileNotFoundError("No template found for refining questions. Ensure file with starts with `refine_question_xxx` exists to evolve/refine the questions.")
    return template_paths

def evolve_question(
        original_input:str, 
        contexts:list[str], 
        generator_model:str, 
        api_key:str,
        base_url_generator:str,
        generator_temperature:float=.3, 
        max_tokens:int|None=None,
        steps:int=2
    ):
    template_paths = get_all_refine_question_template()
    curr_input = original_input
    for _ in range(steps):
        choosen_template = get_template(random.choice(template_paths))
        evolved_prompt = choosen_template.render(contexts=contexts, original_input=curr_input)
        curr_input = ChatOpenAI(model=generator_model, base_url=base_url_generator, api_key=api_key, temperature=generator_temperature, max_completion_tokens=max_tokens).invoke(evolved_prompt)
    return curr_input

def create_answer(
        question:str, 
        contexts:list[str], 
        generator_model:str,
        api_key:str, 
        base_url_generator:str,
        generator_temperature:float=.3, 
        max_tokens:int|None=None
        ):
    prompt = get_template('expected-output-prompt.jinja').render(original_input=question, contexts=contexts)
    expected_output = ChatOpenAI(model=generator_model, base_url=base_url_generator, api_key=api_key, temperature=generator_temperature, max_completion_tokens=max_tokens).invoke(prompt)
    return expected_output


def validate_embedding_api_key(ctx, param, value):
    base_url = ctx.params.get('base_url_embedder', 'http://localhost:8000/v1')

    is_localhost = 'localhost' in base_url or '127.0.0.1' in base_url

    if not is_localhost and value is None:
        value = click.prompt('API key for embedding model', hide_input=True, confirmation_prompt=True, default='')

    return value if value else None

def validate_generator_api_key(ctx, param, value):
    base_url = ctx.params.get('base_url_generator', 'http://localhost:8000/v1')

    is_localhost = 'localhost' in base_url or '127.0.0.1' in base_url

    if not is_localhost and value is None:
        value = click.prompt('API key for generator model', hide_input=True, confirmation_prompt=True, default='')

    return value if value else None

@click.command()
@click.option('--file', help='File path context for creating a question and answer. The question and answer will be based on the content of this file.')
@click.option('--generator_model', help='Used model to generate question and answer. Pass the openrouter model name')
@click.option('--base_url_generator', default='https://openrouter.ai/api/v1', help='Base url for generator model, default to `https://openrouter.ai/api/v1`')
@click.option('--embedder_model', help='Embedder model to embed and get the relevant chunks as a context for generating the question and answer.')
@click.option('--base_url_embedder', default='http://localhost:8000/v1', help='Base url ONLY FOR EMBEDDER MODEL, default to `http://localhost:8000/v1`')
@click.option('--epoch', default=3, help='Number of epoch you wish to generate. Number of samples cant be predicted due to random number of context')
@click.option('--question_evolve_steps', default=2, help='Number of steps to evolve questions using generator model.')
# @click.password_option('--api_key_generator', help='Prompt up API KEY for generator model API.')
@click.option('--api_key_generator', default=None, help='API KEY for generator model API (only needed if using non-localhost)', callback=validate_generator_api_key)
# @click.password_option('--api_key_embedding', default=None, help='Prompt up API KEY for embedding model API.')
@click.option('--api_key_embedding', default=None, help='API KEY for embedding model API (only needed if using non-localhost)', callback=validate_embedding_api_key)
def run(file:str, generator_model:str, base_url_generator:str, embedder_model:str, base_url_embedder:str, epoch:int, question_evolve_steps:int, api_key_generator:str, api_key_embedding:str|None):
    nlp = check_spacy_deps()
    pdfReader = DoclingPdfReader()
    chunker = WordAwareLegalSplitter(nlp=nlp)
    text = pdfReader.extract(file)
    raw_chunk = chunker.split_text(text)
    LOGGER.info(f'Embed-ing {len(raw_chunk)} chunks')
    embeddings = embed_chunks(raw_chunk, base_url_embedder=base_url_embedder, model=embedder_model, api_key=api_key_embedding or 'dummykey')
    LOGGER.info(f'Entering loop with {epoch} epoch')
    synth_data = []
    for ep in tqdm(range(epoch), desc='Generating data'):
        contexts = relevant_context_generator(embeddings=embeddings)
        tqdm.write(f'Found {len(contexts)} context...')
        tqdm.write(f'Creating questions with {generator_model}...')
        questions = create_questions(contexts=contexts, generator_model=generator_model, generator_base_url=base_url_generator, api_key=api_key_generator or 'dummykey')
        for i, question in enumerate(questions):
            tqdm.write(f'Processing question number {i+1}')
            evolved_question = evolve_question(original_input=question, contexts=contexts, generator_model=generator_model, api_key=api_key_generator or 'dummykey', base_url_generator=base_url_generator, steps=question_evolve_steps)
            answer = create_answer(question=evolved_question, contexts=contexts, generator_model=generator_model, api_key=api_key_generator or 'dummykey', base_url_generator=base_url_generator)
            synth_data.append({
                'question': evolved_question.content,
                'answer': answer.content
            })
    
    output_filename = uuid.uuid4().hex + '_' + datetime.now().strftime('%d-%m-%Y') + '.json'
    LOGGER.info(f'Writing synth data to {output_filename}')
    with open(output_filename, 'w') as f:
        json.dump(synth_data, f, indent=4)

if __name__ == "__main__":
    run()