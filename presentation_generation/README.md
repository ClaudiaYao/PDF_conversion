# Presentation Generation

This generate_ppt.ipynb is responsible for generating a final PowerPoint presentation, taking in the summarized section-level text as input. The output is saved as a .pptx file.

## Customization

For the parsing of the summarized text, there are two options: using a GPT model and manual.

1. For the GPT model, `Pydantic` `BaseObjects` are defined to reflect the schema and format of the desired JSON output from the LLM. A LangChain chain is constructed with a query and the Pydantic parser to ensure the output matches the desired JSON format. Both the query and `BaseObjects` may be edited, enhanced to include additional elements of the PowerPoint.
2. For the manual process, the format and layout of each slide should be manually defined. By default, each sentence will form a new bullet point, and each slide represents one section.

## Note

1. The final Powerpoints are saved in the "data/powerpoints" sub folder.
2. You would need to provide an OpenAI API key in a .env file in order to use the GPT model for parsing.
3. You could also provide a Powerpoint with an existing theme in the "data/ppt_themes" folder. The generated powerpoint can be set to follow that theme.
