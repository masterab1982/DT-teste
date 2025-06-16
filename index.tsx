import {
  GoogleGenAI,
  HarmCategory,
  HarmBlockThreshold,
  Content,
  Chat,
  GenerateContentResponse,
  Part,
} from "@google/genai";
import { marked } from "marked";

// تأكد من معالجة مفتاح API وفقًا للإرشادات
const API_KEY = process.env.API_KEY;

const errorMessageGlobalDiv = document.getElementById('error-message') as HTMLDivElement;
const suggestedQuestionsContainer = document.getElementById('suggested-questions-container') as HTMLDivElement;


if (!API_KEY) {
  console.error("API_KEY is not set. Please set the API_KEY environment variable.");
  if (errorMessageGlobalDiv) {
    errorMessageGlobalDiv.textContent = "API_KEY is not configured. Please check the console.";
    errorMessageGlobalDiv.style.display = 'block';
  }
  if (suggestedQuestionsContainer) {
    suggestedQuestionsContainer.style.display = 'none';
  }
  throw new Error("API_KEY not set");
}

const ai = new GoogleGenAI({ apiKey: API_KEY });

const modelName = 'gemini-2.5-flash-preview-04-17';
  
const safetySettings = [
  { category: HarmCategory.HARM_CATEGORY_HARASSMENT, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
  { category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold: HarmBlockThreshold.BLOCK_LOW_AND_ABOVE },
];

const systemInstructionTextRAG = `أنت مساعد خبير، متخصص في استراتيجية التحول الرقمي لهيئة الهلال الأحمر السعودي. تجيب على الأسئلة بدقة بناءً على هذه الاستراتيجية. يجب أن تكون إجاباتك طبيعية، كأنك خبيرٌ يمتلك هذه المعرفة بشكل مباشر وأصيل. **ممنوع منعاً باتاً** الإشارة في ردودك إلى أنك تستمد المعلومات من 'وثيقة'، 'سياق'، 'مصدر'، أو أن المعلومات كانت بتنسيق JSON. هدفك هو تقديم إجابة واضحة ومباشرة. إذا كانت الإجابة تتضمن نقاطًا متعددة، استخدم قائمة نقطية لتنظيمها بشكل جيد.`;

interface KnowledgeEntry {
  prompt: string; // Original prompt text
  completion: string;
  sourcePath?: string; // Optional: for debugging or tracing the origin of the entry
  [key: string]: any; 
}
let knowledgeBase: KnowledgeEntry[] = [];
const MAX_CANDIDATES_TO_SEND = 1; // إرسال أفضل تطابق فقط لتركيز النموذج

const visionQueryPattern = /كيف (يساهم|تساهم)(?: مستهدفات)? مشروع (.*?) في تحقيق رؤية (?:المملكة )?(?:2030|٢٠٣٠|2023|٢٠٢٣)(?:م)?\??/i;
const objectiveVisionQueryPattern = /^(?:كيف يساهم|مساهمة|مواءمة) هدف (.*?) (?:في التحول الرقمي )?(?:في تحقيق|مع) رؤية (?:المملكة )?(?:2030|٢٠٣٠|2023|٢٠٢٣)(?:م)?\??$/i;
let userInput = ''; 

async function fetchFileContent(filePath: string): Promise<string> {
  const response = await fetch(filePath);
  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status} while fetching ${filePath}`);
  }
  return await response.text();
}


function transformMetadataToKnowledgeEntriesRecursive(
  data: any,
  path: string[] = [],
  titleHierarchy: string[] = []
): KnowledgeEntry[] {
  let entries: KnowledgeEntry[] = [];
  const currentPathString = path.join(".");

  if (typeof data === 'object' && data !== null) {
    let promptText = "";
    const currentObjectTitle = data.title && typeof data.title === 'string' ? data.title.trim() : "";
    const parentContextTitles = titleHierarchy.filter(t => t && t.trim()).join(" - ");

    if (currentObjectTitle) {
      promptText = `ما هي المعلومات حول "${currentObjectTitle}"` + (parentContextTitles ? ` (ضمن "${parentContextTitles}")` : "") + `؟`;
    } else if (path.length > 0) {
      const lastKey = path[path.length - 1];
      promptText = `ما هي تفاصيل "${lastKey}"` + (parentContextTitles ? ` في قسم "${parentContextTitles}"` : "") + `؟`;
    } else {
      promptText = "ما هي المعلومات العامة عن وثيقة استراتيجية التحول الرقمي؟";
    }
    
    entries.push({
      prompt: promptText,
      completion: JSON.stringify(data, null, 2),
      sourcePath: currentPathString || "root"
    });

    const newTitleHierarchy = currentObjectTitle ? [...titleHierarchy, currentObjectTitle] : titleHierarchy;

    if (Array.isArray(data)) {
      data.forEach((item, index) => {
        const itemIdentifier = (typeof item === 'object' && item !== null && (item.title || item.name || item.category || item.id)) || `البند ${index + 1}`;
        if (typeof item === 'object' && item !== null) {
            entries.push(
              ...transformMetadataToKnowledgeEntriesRecursive(
                item,
                [...path, `[${index}]`],
                currentObjectTitle ? [currentObjectTitle, `عنصر ${itemIdentifier}`] : (path.length > 0 ? [path[path.length - 1], `عنصر ${itemIdentifier}`] : [`عنصر ${itemIdentifier}`])
              )
            );
        }
      });
    } else { 
      for (const key in data) {
        if (data.hasOwnProperty(key)) {
           if (key === 'title' && typeof data[key] === 'string' && currentObjectTitle === data[key]) {
               continue;
           }
          entries.push(
            ...transformMetadataToKnowledgeEntriesRecursive(
              data[key],
              [...path, key],
              newTitleHierarchy
            )
          );
        }
      }
    }
  } else if (path.length > 0) { 
     const lastKey = path[path.length - 1];
     if (lastKey !== 'title' && lastKey !== 'name' && lastKey !== 'id' && lastKey !== 'category') { 
        const parentContextTitles = titleHierarchy.filter(t => t && t.trim()).join(" - ");
        let primitivePrompt = `ما هي قيمة "${lastKey}"`;
        if (parentContextTitles) {
            primitivePrompt += ` في سياق "${parentContextTitles}"`;
        }
        primitivePrompt += `؟`;
        
        entries.push({
            prompt: primitivePrompt,
            completion: JSON.stringify(data, null, 2),
            sourcePath: currentPathString
        });
     }
  }
  return entries;
}

async function transformMetadataToKnowledgeBase(metadataObject: any): Promise<KnowledgeEntry[]> {
  const strategyData = metadataObject.digitalTransformationStrategy || metadataObject;
  const rawEntries = transformMetadataToKnowledgeEntriesRecursive(strategyData);
  
  const uniqueEntriesMap = new Map<string, KnowledgeEntry>(); 
  
  for (const entry of rawEntries) {
    if (entry.completion === "null" || 
        entry.completion === "{}" || 
        entry.completion === "[]" || 
        entry.completion === "\"\"") {
      continue;
    }
    
    const normalizedKey = entry.prompt.trim().toLowerCase().replace(/[؟.,!]/g, '');
    if (!uniqueEntriesMap.has(normalizedKey)) {
      uniqueEntriesMap.set(normalizedKey, entry); 
    }
  }

  const digitalStrategy = metadataObject.digitalTransformationStrategy || metadataObject;

  // Vision, Mission, and Pillars specific entries
  const strategicHouse = digitalStrategy.strategicHouse;
  if (strategicHouse) {
      if (strategicHouse.vision && typeof strategicHouse.vision === 'string') {
          const visionText = strategicHouse.vision;
          const visionPrompts = [
              "ما هي رؤية التحول الرقمي؟", "رؤية التحول الرقمي", "ماهي الرؤية الاستراتيجية للتحول الرقمي؟",
              "اذكر لي رؤية التحول الرقمي", "رؤية الهيئة للتحول الرقمي", "ما هي رؤية الهلال الأحمر للتحول الرقمي؟",
              "رؤية التحول الرقمي للهيئة",
              "ماهي رؤية استراتيجية التحول الرقمي" 
          ];
          // More explicit completion for vision questions
          const visionCompletion = `رؤية التحول الرقمي لهيئة الهلال الأحمر السعودي هي: "${visionText}"`;
          visionPrompts.forEach(promptText => {
              const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
              uniqueEntriesMap.set(normalizedKey, {
                  prompt: promptText, completion: visionCompletion, sourcePath: "digitalTransformationStrategy.strategicHouse.vision"
              });
          });
      }

      if (strategicHouse.mission && typeof strategicHouse.mission === 'string') {
          const missionText = strategicHouse.mission;
          const missionPrompts = [
              "ما هي رسالة التحول الرقمي؟", "رسالة التحول الرقمي", "ماهي الرسالة الاستراتيجية للتحول الرقمي؟",
              "اذكر لي رسالة التحول الرقمي", "رسالة الهيئة للتحول الرقمي", "ما هي رسالة الهلال الأحمر للتحول الرقمي؟",
              "رسالة التحول الرقمي للهيئة"
          ];
           // More explicit completion for mission questions
          const missionCompletion = `رسالة التحول الرقمي لهيئة الهلال الأحمر السعودي هي: "${missionText}"`;
          missionPrompts.forEach(promptText => {
              const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
              uniqueEntriesMap.set(normalizedKey, {
                  prompt: promptText, completion: missionCompletion, sourcePath: "digitalTransformationStrategy.strategicHouse.mission"
              });
          });
      }
      
      // Explicit entry for "اذكر الركائز الاستراتيجية للتحول الرقمي"
      if (strategicHouse.pillarsData && strategicHouse.pillarsData.pillars && Array.isArray(strategicHouse.pillarsData.pillars)) {
          const pillars = strategicHouse.pillarsData.pillars;
          let pillarsListText = "الركائز الاستراتيجية للتحول الرقمي هي:\n";
          pillars.forEach((pillar: any) => {
              pillarsListText += `- ${pillar.name}`;
              // Optional: Add description if needed for more detailed questions, but keep concise for listing.
              // if (pillar.description) {
              //     pillarsListText += `: ${pillar.description}`;
              // }
              pillarsListText += "\n";
          });

          const pillarListPrompts = [
              "اذكر الركائز الاستراتيجية للتحول الرقمي.",
              "ما هي الركائز الاستراتيجية للتحول الرقمي؟",
              "عدد الركائز الاستراتيجية للتحول الرقمي.",
              "قائمة الركائز الاستراتيجية للتحول الرقمي",
              "ما هي ركائز التحول الرقمي؟"
          ];
          pillarListPrompts.forEach(promptText => {
              const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
              uniqueEntriesMap.set(normalizedKey, {
                  prompt: promptText,
                  completion: pillarsListText.trim(),
                  sourcePath: "digitalTransformationStrategy.strategicHouse.pillarsData.pillars.list"
              });
          });
      }
  }


    if (digitalStrategy.strategicHouse &&
      digitalStrategy.strategicHouse.pillarsData &&
      digitalStrategy.strategicHouse.pillarsData.pillars &&
      Array.isArray(digitalStrategy.strategicHouse.pillarsData.pillars) &&
      digitalStrategy.strategicHouse.objectivesData &&
      digitalStrategy.strategicHouse.objectivesData.objectives &&
      Array.isArray(digitalStrategy.strategicHouse.objectivesData.objectives)) {

      const pillars = digitalStrategy.strategicHouse.pillarsData.pillars;
      const objectives = digitalStrategy.strategicHouse.objectivesData.objectives;
      
      let pillarsCompletionText = ""; 
      if (digitalStrategy.strategicHouse.title) {
          pillarsCompletionText += `## ${digitalStrategy.strategicHouse.title}\n\n`;
      }
      pillarsCompletionText += "فيما يلي ربط الركائز الاستراتيجية للتحول الرقمي بأهدافها المقابلة:\n\n";

      for (const pillar of pillars) {
          pillarsCompletionText += `### ركيزة: ${pillar.name}\n`;
          if (pillar.description) {
              pillarsCompletionText += `**الوصف:** ${pillar.description}\n`;
          }
          const relatedObjectives = objectives.filter(obj => obj.pillar === pillar.name);
          if (relatedObjectives.length > 0) {
              pillarsCompletionText += `**الأهداف الاستراتيجية المرتبطة بهذه الركيزة:**\n`;
              for (const obj of relatedObjectives) {
                  pillarsCompletionText += `- **${obj.id}**: ${obj.name}\n`;
              }
          } else {
              pillarsCompletionText += "- لا توجد أهداف استراتيجية محددة مرتبطة مباشرة بهذه الركيزة ضمن البيانات المتوفرة.\n";
          }
          pillarsCompletionText += "\n"; 
      }

      const promptsForPillarsLink = [
          "اربط الركائز بالاهداف الاستراتيجية للتحول الرقمي",
          "ما هي العلاقة بين الركائز الاستراتيجية والأهداف الاستراتيجية للتحول الرقمي؟",
          "كيف ترتبط ركائز التحول الرقمي بأهدافها الاستراتيجية؟",
          "اعرض لي الأهداف الاستراتيجية لكل ركيزة من ركائز التحول الرقمي.",
          "ما هي الأهداف التابعة لكل ركيزة استراتيجية؟",
          "ربط الركائز بالأهداف الاستراتيجية",
          "الركائز والأهداف الاستراتيجية المرتبطة بها",
          "ما هي الاهداف الاستراتيجية للتحول الرقمي وارتباطها بالركائز الاستراتيجية"
      ];

      promptsForPillarsLink.forEach(promptText => {
          const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
          uniqueEntriesMap.set(normalizedKey, { 
              prompt: promptText, 
              completion: pillarsCompletionText, 
              sourcePath: "digitalTransformationStrategy.strategicHouse.pillars_objectives_link" 
          });
      });
  }

  const roadmapData = digitalStrategy.roadmap;
  const allProjectsList = digitalStrategy.futureProjects?.projects;

  if (roadmapData && roadmapData.timeline && Array.isArray(roadmapData.timeline) && allProjectsList && Array.isArray(allProjectsList)) {
      for (const timelineEntry of roadmapData.timeline) {
          if (timelineEntry.year && timelineEntry.projects && Array.isArray(timelineEntry.projects)) {
              const yearName = timelineEntry.year; 
              const yearProjectIds = timelineEntry.projects as string[];
              
              const yearProjectsDetailsText = yearProjectIds.map(id => {
                  const projectDetail = allProjectsList.find(p => p.id === id);
                  return projectDetail ? `- ${projectDetail.name} (المعرف: ${projectDetail.id})` : `- مشروع بالمعرف ${id} (تفاصيل الاسم غير متوفرة في قائمة المشاريع المفصلة)`;
              }).join("\n");

              if (yearProjectsDetailsText.length === 0 && yearProjectIds.length > 0) {
                continue; 
              }
              
              let yearCompletionText = `## مشاريع ${yearName}\n\n`;
              if (yearProjectIds.length > 0) {
                yearCompletionText += `في ${yearName}، المشاريع المخطط لها هي:\n${yearProjectsDetailsText}\n\n`;
              } else {
                yearCompletionText += `لا توجد مشاريع محددة لـ ${yearName} ضمن البيانات المتوفرة.\n\n`;
              }

              if(timelineEntry.cost) {
                  yearCompletionText += `**التكلفة الإجمالية المقدرة لمشاريع ${yearName}:** ${timelineEntry.cost} ر.س.\n`;
              }
              if(typeof timelineEntry.projectCount === 'number') { 
                  yearCompletionText += `**إجمالي عدد المشاريع في ${yearName}:** ${timelineEntry.projectCount}.\n`;
              }
              
              const yearNumberMatch = yearName.match(/\((\d{4})\)/);
              const yearNumberStr = yearNumberMatch ? yearNumberMatch[1] : null;

              const yearSpecificPrompts: string[] = [];
              yearSpecificPrompts.push(`ما هي مشاريع ${yearName}؟`);
              yearSpecificPrompts.push(`مشاريع ${yearName}`);
              yearSpecificPrompts.push(`اذكر لي مشاريع ${yearName}`);
              yearSpecificPrompts.push(`ما هي خطة المشاريع لـ ${yearName}؟`);
              yearSpecificPrompts.push(`تفاصيل مشاريع ${yearName}`);


              if (yearNumberStr) {
                  yearSpecificPrompts.push(`ما هي مشاريع سنة ${yearNumberStr}؟`);
                  yearSpecificPrompts.push(`مشاريع سنة ${yearNumberStr}`);
                  yearSpecificPrompts.push(`مشاريع عام ${yearNumberStr}`);
                  yearSpecificPrompts.push(`خطة مشاريع ${yearNumberStr}`);
                  yearSpecificPrompts.push(`تفاصيل مشاريع سنة ${yearNumberStr}`);
                  
                  if (yearName.toLowerCase().includes("السنة الاولى")) {
                      yearSpecificPrompts.push(`ما هي مشاريع السنة الاولى ${yearNumberStr}؟`);
                      yearSpecificPrompts.push(`مشاريع السنة الاولى عام ${yearNumberStr}`);
                  } else if (yearName.toLowerCase().includes("السنة الثانية")) {
                      yearSpecificPrompts.push(`ما هي مشاريع السنة الثانية ${yearNumberStr}؟`);
                      yearSpecificPrompts.push(`مشاريع السنة الثانية عام ${yearNumberStr}`);
                  } else if (yearName.toLowerCase().includes("السنة الثالثة")) {
                       yearSpecificPrompts.push(`ما هي مشاريع السنة الثالثة ${yearNumberStr}؟`);
                       yearSpecificPrompts.push(`مشاريع السنة الثالثة عام ${yearNumberStr}`);
                  }
              }
              
              if (yearName.toLowerCase().includes("السنة الاولى")) {
                   yearSpecificPrompts.push(`ما هي مشاريع السنة الاولى؟`);
                   yearSpecificPrompts.push(`مشاريع السنة الاولى`);
              } else if (yearName.toLowerCase().includes("السنة الثانية")) {
                   yearSpecificPrompts.push(`ما هي مشاريع السنة الثانية؟`);
                   yearSpecificPrompts.push(`مشاريع السنة الثانية`);
              } else if (yearName.toLowerCase().includes("السنة الثالثة")) {
                   yearSpecificPrompts.push(`ما هي مشاريع السنة الثالثة؟`);
                   yearSpecificPrompts.push(`مشاريع السنة الثالثة`);
              }
              
              const uniqueYearPrompts = Array.from(new Set(yearSpecificPrompts)); 

              uniqueYearPrompts.forEach(promptText => {
                  const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
                  uniqueEntriesMap.set(normalizedKey, {
                      prompt: promptText, 
                      completion: yearCompletionText,
                      sourcePath: `digitalTransformationStrategy.roadmap.timeline.${yearName}`
                  });
              });
          }
      }
  }

  const devMethodology = digitalStrategy.developmentMethodology;
  if (devMethodology && devMethodology.introduction && typeof devMethodology.introduction === 'string' &&
      devMethodology.steps && Array.isArray(devMethodology.steps)) {
      
      let methodologyCompletionText = `## ${devMethodology.title || 'منهجية تطوير استراتيجية التحول الرقمي'}\n\n`;
      methodologyCompletionText += `${devMethodology.introduction}\n\n`;
      
      if (devMethodology.steps.length > 0) {
        methodologyCompletionText += "تتكون المنهجية المتبعة من عدة خطوات رئيسية، وهي كالتالي:\n\n";
        devMethodology.steps.forEach((step: any) => {
            methodologyCompletionText += `### الخطوة ${step.step}: ${step.title}\n`;
            if (step.description) methodologyCompletionText += `**الوصف:** ${step.description}\n`;
            if (step.details) methodologyCompletionText += `**التفاصيل:** ${step.details}\n`;
            methodologyCompletionText += "\n";
        });
      }

      const methodologyPrompts = [
          "كيف تم بناء استراتيجية التحول الرقمي؟", "ماهي المنهجية المتبعة لبناء استراتيجية التحول الرقمي؟",
          "منهجية بناء الاستراتيجية", "اشرح منهجية تطوير استراتيجية التحول الرقمي.",
          "ما هي خطوات بناء استراتيجية التحول الرقمي؟", "منهجية أعمال تطوير استراتيجية التحول الرقمي",
          "كيف وضعتم استراتيجية التحول الرقمي؟", "ما هي منهجية تطوير الاستراتيجية؟",
          "صف لنا منهجية بناء الاستراتيجية", "خطوات تطوير الاستراتيجية الرقمية",
          "المنهجية المتبعة لتطوير استراتيجية التحول الرقمي", "ما هي آلية بناء استراتيجية التحول الرقمي؟"
      ];
      methodologyPrompts.forEach(promptText => {
          const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
          uniqueEntriesMap.set(normalizedKey, {
              prompt: promptText, completion: methodologyCompletionText, sourcePath: "digitalTransformationStrategy.developmentMethodology.summary"
          });
      });
  }

  const allProjects = digitalStrategy.futureProjects?.projects;
  const allInitiatives = digitalStrategy.digitalTransformationInitiatives?.initiativeDetails;
  const allGapDomains = digitalStrategy.gapAnalysis?.domains;
  const allStrategicObjectives = digitalStrategy.strategicHouse?.objectivesData?.objectives;
  const allKpis = digitalStrategy.performanceIndicators?.kpis;
  const projectPriorities = digitalStrategy.projectPrioritization?.priorities;

  const domainToPillarMapping: { [key: string]: string[] } = {
    "التطبيقات الرقمية وتحسين تجربة العميل": ["تجربة رقمية فريدة", "حلول ابتكارية"],
    "البنية التقنية وأمن المعلومات": ["بيئة موثوقة"],
    "قدرات الاعمال (الخدمات)": ["منظومة تشغيلية متميزة"],
    "حوكمة التحول الرقمي": ["منظومة تشغيلية متميزة"],
    "البيانات وذكاء الاعمال": ["منظومة تشغيلية متميزة", "حلول ابتكارية", "بيئة موثوقة"],
    "التقنيات الناشئة": ["حلول ابتكارية"]
  };

  if (allProjects && Array.isArray(allProjects)) {
    for (const project of allProjects) {
      if (!project.id || !project.name) continue;

      let projectCompletionText = `## تفاصيل مشروع: ${project.name} (المعرف: ${project.id})\n\n`;
      projectCompletionText += `**المشروع:** ${project.name}\n`;
      projectCompletionText += `**المعرف:** ${project.id}\n`;
      if (project.cost_sar) {
        projectCompletionText += `**التكلفة المقدرة:** ${project.cost_sar} ريال سعودي\n`;
      }
      if (project.duration_months) {
        projectCompletionText += `**مدة التنفيذ المقدرة:** ${project.duration_months} شهرًا\n`;
      }
      
      let yearNameForProject: string | null = null;
      if (roadmapData && roadmapData.timeline && Array.isArray(roadmapData.timeline)) {
        for (const timelineEntry of roadmapData.timeline) {
          if (timelineEntry.projects && Array.isArray(timelineEntry.projects) && timelineEntry.projects.includes(project.id)) {
            yearNameForProject = timelineEntry.year;
            break; 
          }
        }
      }

      if (yearNameForProject) {
        projectCompletionText += `\n**سنة التنفيذ المخطط لها:** ${yearNameForProject}.\n`;
        projectCompletionText += `سيتم تنفيذ هذا المشروع ضمن خطة ${yearNameForProject}.\n\n`;
      }
      
      const parentInitiativeDetail = allInitiatives?.find(init => init.name === project.initiative);
      
      if (parentInitiativeDetail) {
        projectCompletionText += `\n### المبادرة الأم: ${parentInitiativeDetail.name}\n`;
        if (parentInitiativeDetail.description) {
          projectCompletionText += `**وصف وأهداف المبادرة:** ${parentInitiativeDetail.description}\n`;
        }
      } else if (project.initiative) {
        projectCompletionText += `\n**المبادرة الأم:** ${project.initiative} (لم يتم العثور على تفاصيل إضافية لهذه المبادرة).\n`;
      }
      projectCompletionText += "\n";

      const initiativeDomain = parentInitiativeDetail?.domain;
      const relatedPillarNames = initiativeDomain ? domainToPillarMapping[initiativeDomain] : [];
      const relevantStrategicObjectives: any[] = [];

      if (relatedPillarNames && relatedPillarNames.length > 0 && allStrategicObjectives && Array.isArray(allStrategicObjectives)) {
        for (const pillarName of relatedPillarNames) {
          const objectivesInPillar = allStrategicObjectives.filter(obj => obj.pillar === pillarName);
          relevantStrategicObjectives.push(...objectivesInPillar);
        }
      }
      
      const uniqueStrategicObjectives = Array.from(new Set(relevantStrategicObjectives.map(obj => obj.id)))
                                          .map(id => relevantStrategicObjectives.find(obj => obj.id === id))
                                          .filter(obj => obj);

      if (uniqueStrategicObjectives.length > 0) {
        projectCompletionText += `### الأهداف الاستراتيجية (للتحول الرقمي) المرتبطة:\n`;
        projectCompletionText += `يساهم هذا المشروع، من خلال مبادرته الأم "${parentInitiativeDetail?.name || project.initiative}", في تحقيق الأهداف الاستراتيجية التالية للتحول الرقمي:\n`;
        for (const obj of uniqueStrategicObjectives) {
          projectCompletionText += `- **${obj.id} ${obj.name}** (الركيزة: ${obj.pillar})\n`;
        }
        projectCompletionText += "\n";

        if (allKpis && Array.isArray(allKpis)) {
          const relevantKpis: any[] = [];
          for (const stratObj of uniqueStrategicObjectives) {
            const kpisForObjective = allKpis.filter(kpi => kpi.relatedObjective === stratObj.name);
            relevantKpis.push(...kpisForObjective);
          }
          const uniqueKpis = Array.from(new Set(relevantKpis.map(kpi => kpi.id)))
                                 .map(id => relevantKpis.find(kpi => kpi.id === id))
                                 .filter(kpi => kpi); 

          if (uniqueKpis.length > 0) {
            projectCompletionText += `**المؤشرات الاستراتيجية المرتبطة بهذه الأهداف:**\n`;
            for (const kpi of uniqueKpis) {
              projectCompletionText += `- **${kpi.id}:** ${kpi.name}\n`;
            }
            projectCompletionText += "\n";
          }
        }
      }
      
      if (projectPriorities && Array.isArray(projectPriorities)) {
        const priorityInfo = projectPriorities.find(p => p.id === project.id);
        if (priorityInfo && priorityInfo.priority) {
          projectCompletionText += `**درجة أهمية المشروع (الأولوية):** ${priorityInfo.priority}\n\n`;
        }
      }

      if (parentInitiativeDetail && allGapDomains && Array.isArray(allGapDomains)) {
        let gapsBridgedText = "";
        for (const domain of allGapDomains) {
          if (domain.gaps && Array.isArray(domain.gaps)) {
            for (const gap of domain.gaps) {
              if (gap.bridgingInitiative && typeof gap.bridgingInitiative === 'string' &&
                  (gap.bridgingInitiative === parentInitiativeDetail.name || gap.bridgingInitiative.includes(parentInitiativeDetail.name))) {
                if (gapsBridgedText === "") {
                     gapsBridgedText += `يهدف هذا المشروع، من خلال مبادرته الأم "${parentInitiativeDetail.name}", إلى معالجة التحديات والمشاكل (الفجوات) التالية:\n\n`;
                }
                gapsBridgedText += `#### تحدي/مشكلة (فجوة): ${gap.description}\n`;
                if (gap.impact) {
                  gapsBridgedText += `- **التأثير السلبي الحالي:** ${gap.impact}\n`;
                }
                if (gap.futureState) {
                  gapsBridgedText += `- **الوضع المستهدف بعد المعالجة:** ${gap.futureState}\n`;
                }
                gapsBridgedText += "\n";
              }
            }
          }
        }
        if (gapsBridgedText) {
           projectCompletionText += `### التحديات والمشاكل التي يعالجها المشروع (من خلال سد الفجوات عبر المبادرة الأم):\n${gapsBridgedText}`;
        }
      }
      
      // حقل اختياري جديد في metadata.json لكل مشروع
      if (project.vision2030Alignment && Array.isArray(project.vision2030Alignment) && project.vision2030Alignment.length > 0) {
        projectCompletionText += `\n### المساهمة في تحقيق رؤية المملكة 2030:\n`; 
        projectCompletionText += `يساهم هذا المشروع في تحقيق أهداف رؤية المملكة 2030 من خلال النقاط التالية:\n`; 
        project.vision2030Alignment.forEach((alignment: any) => {
          if (alignment.visionObjective && alignment.projectContribution) {
            projectCompletionText += `- **${alignment.visionObjective.trim()}:** ${alignment.projectContribution.trim()}\n`;
          }
        });
        projectCompletionText += "\n";
      } else if (userInput.match(visionQueryPattern)) { 
          projectCompletionText += `\n*لم يتم تحديد مساهمات مباشرة لهذا المشروع في أهداف رؤية المملكة 2030 ضمن البيانات المتوفرة محليًا. قد يتم البحث عن هذه المعلومة عبر الإنترنت إذا كان السؤال يتعلق بذلك.*\n`;
      }


      const projectSpecificPrompts: string[] = [
        `ما هي تفاصيل مشروع "${project.name}"؟`, `معلومات عن مشروع "${project.name}"`,
        `حدثني عن مشروع "${project.name}"`, `ما هو مشروع "${project.name}" (المعرف ${project.id})؟`,
        `تفاصيل ${project.id}`, `مشروع ${project.id}`, `"${project.name}"`,
        `ما هي أهداف مشروع "${project.name}"؟`, 
        `ما هو تأثير مشروع "${project.name}"؟`,
        `ما هو الأثر من تنفيذ مشروع "${project.name}"؟`, `ماذا يحقق مشروع "${project.name}"؟`,
        `ما هي الأهداف الاستراتيجية المرتبطة بمشروع "${project.name}"؟`,
        `اذكر الأهداف الاستراتيجية لمشروع "${project.name}"`,
        `ما هي المؤشرات الاستراتيجية المرتبطة بمشروع "${project.name}"؟`,
        `ما هي مؤشرات الأداء المتأثرة بمشروع "${project.name}"؟`,
        `اذكر المؤشرات الاستراتيجية المرتبطة والمتأثرة بمشروع "${project.name}"`,
        `ما هي درجة أهمية مشروع "${project.name}"؟`,
        `ما مدى أهمية مشروع "${project.name}"؟`,
        `كيف تساهم مستهدفات مشروع ${project.name} في تحقيق رؤية المملكة 2030م؟`,
        `كيف يساهم مشروع ${project.name} في تحقيق رؤية المملكة 2030؟`,
        `مساهمة مشروع ${project.name} في رؤية 2030`,
        `كيف يدعم مشروع ${project.name} رؤية المملكة 2030؟`,
        `ما هي التحديات التي يعالجها مشروع "${project.name}"؟`,
        `ما هي المشاكل التي يحلها مشروع "${project.name}"؟`,
        `ما هي التحديات أو المشاكل المتوقع معالجتها في حال تمت الموافقة على مشروع "${project.name}"؟`,
        `التحديات التي يتصدى لها مشروع "${project.name}"`,
        `المشاكل التي يعالجها مشروع "${project.name}"`
      ];

      if (yearNameForProject) {
        projectSpecificPrompts.push(`في أي سنة سيتم تنفيذ مشروع "${project.name}"؟`);
        projectSpecificPrompts.push(`متى سيتم تنفيذ مشروع "${project.name}"؟`);
        projectSpecificPrompts.push(`ما هي سنة تنفيذ مشروع "${project.name}"؟`);
        projectSpecificPrompts.push(`جدول تنفيذ مشروع "${project.name}"`);
        const yearNumberMatchLocal = yearNameForProject.match(/\((\d{4})\)/);
        const yearNumberStrLocal = yearNumberMatchLocal ? yearNumberMatchLocal[1] : null;
        if (yearNumberStrLocal) {
            projectSpecificPrompts.push(`في أي عام سيبدأ مشروع "${project.name}"؟`);
            projectSpecificPrompts.push(`في أي عام (${yearNumberStrLocal}) يخطط لتنفيذ مشروع "${project.name}"؟`);
        }
      }
      
      const projectNameParts = project.name.split(/[\s-]+/);
      if (projectNameParts.length > 1) {
          const lastPart = projectNameParts[projectNameParts.length - 1];
          if (lastPart && lastPart.length > 2 && !/^\d+$/.test(lastPart) && project.name.toLowerCase().includes(lastPart.toLowerCase())) {
             if (lastPart.toLowerCase() !== project.name.toLowerCase()){
                projectSpecificPrompts.push(`ما هو مشروع "${lastPart}"؟`);
                projectSpecificPrompts.push(`تفاصيل مشروع "${lastPart}"`);
                projectSpecificPrompts.push(`أهداف مشروع "${lastPart}"`);
                projectSpecificPrompts.push(`تأثير مشروع "${lastPart}"`);
                projectSpecificPrompts.push(`الأهداف الاستراتيجية لمشروع "${lastPart}"`);
                projectSpecificPrompts.push(`درجة أهمية مشروع "${lastPart}"`);
                 if (yearNameForProject) {
                    projectSpecificPrompts.push(`في أي سنة سيتم تنفيذ مشروع "${lastPart}"؟`);
                }
                projectSpecificPrompts.push(`كيف تساهم مستهدفات مشروع ${lastPart} في تحقيق رؤية المملكة 2030م؟`);
             }
          }
      }
      if (project.name.includes("مصادر")) { 
            projectSpecificPrompts.push(`ما هو مشروع مصادر؟`, `تفاصيل مشروع مصادر`, `أهداف مشروع مصادر`, `تأثير مشروع مصادر`, `الأهداف الاستراتيجية لمشروع مصادر`, `درجة أهمية مشروع مصادر`);
            projectSpecificPrompts.push(`كيف تساهم مستهدفات مشروع مصادر في تحقيق رؤية المملكة 2030م؟`);
            if (yearNameForProject) {
                projectSpecificPrompts.push(`في أي سنة سيتم تنفيذ مشروع مصادر؟`);
            }
      }
       if (projectPriorities && Array.isArray(projectPriorities)) {
        const priorityInfo = projectPriorities.find(p => p.id === project.id);
        if (priorityInfo && priorityInfo.priority) {
            projectSpecificPrompts.push(`هل مشروع "${project.name}" ذو أولوية ${priorityInfo.priority}؟`);
        }
      }

      Array.from(new Set(projectSpecificPrompts)).forEach(promptText => {
        const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
        uniqueEntriesMap.set(normalizedKey, {
          prompt: promptText, completion: projectCompletionText,
          sourcePath: `digitalTransformationStrategy.futureProjects.projects.${project.id}`
        });
      });
    }
  } 

  // 7. Add/Overwrite with Detailed Initiative-Specific Entries
  if (allInitiatives && Array.isArray(allInitiatives)) {
    for (const initiative of allInitiatives) {
      if (!initiative.id || !initiative.name) continue;

      let initiativeCompletionText = `## تفاصيل مبادرة: ${initiative.name} (المعرف: ${initiative.id})\n\n`;
      initiativeCompletionText += `**المبادرة:** ${initiative.name}\n`;
      initiativeCompletionText += `**المعرف:** ${initiative.id}\n`;
      if (initiative.description) {
        initiativeCompletionText += `**الوصف:** ${initiative.description}\n`;
      }
      if (initiative.domain) {
        initiativeCompletionText += `**المجال:** ${initiative.domain}\n`;
      }
      if (initiative.estimatedCost) {
        initiativeCompletionText += `**التكلفة المقدرة:** ${initiative.estimatedCost}\n`;
      }
      if (typeof initiative.projectCount === 'number') {
        initiativeCompletionText += `**عدد المشاريع التابعة:** ${initiative.projectCount}\n`;
      }

      if (initiative.projects && Array.isArray(initiative.projects) && initiative.projects.length > 0) {
        initiativeCompletionText += `\n### المشاريع التابعة لهذه المبادرة:\n`;
        initiative.projects.forEach((projName: string) => {
          const projectDetail = allProjects?.find(p => p.name === projName);
          initiativeCompletionText += `- ${projName}${projectDetail ? ` (المعرف: ${projectDetail.id})` : ''}\n`;
        });
        initiativeCompletionText += "\n";
      }

      const initiativeDomain = initiative.domain;
      const relatedPillarNames = initiativeDomain ? domainToPillarMapping[initiativeDomain] : [];
      const relevantStrategicObjectives: any[] = [];

      if (relatedPillarNames && relatedPillarNames.length > 0 && allStrategicObjectives && Array.isArray(allStrategicObjectives)) {
        for (const pillarName of relatedPillarNames) {
          const objectivesInPillar = allStrategicObjectives.filter(obj => obj.pillar === pillarName);
          relevantStrategicObjectives.push(...objectivesInPillar);
        }
      }
      
      const uniqueStrategicObjectives = Array.from(new Set(relevantStrategicObjectives.map(obj => obj.id)))
                                          .map(id => relevantStrategicObjectives.find(obj => obj.id === id))
                                          .filter(obj => obj);

      if (uniqueStrategicObjectives.length > 0) {
        initiativeCompletionText += `### الأهداف الاستراتيجية العامة التي تساهم بها المبادرة:\n`;
        for (const obj of uniqueStrategicObjectives) {
          initiativeCompletionText += `- **${obj.id} ${obj.name}** (الركيزة: ${obj.pillar})\n`;
        }
        initiativeCompletionText += "\n";

        if (allKpis && Array.isArray(allKpis)) {
          const relevantKpis: any[] = [];
          for (const stratObj of uniqueStrategicObjectives) {
            const kpisForObjective = allKpis.filter(kpi => kpi.relatedObjective === stratObj.name);
            relevantKpis.push(...kpisForObjective);
          }
          const uniqueKpis = Array.from(new Set(relevantKpis.map(kpi => kpi.id)))
                                 .map(id => relevantKpis.find(kpi => kpi.id === id))
                                 .filter(kpi => kpi); 

          if (uniqueKpis.length > 0) {
            initiativeCompletionText += `**المؤشرات الاستراتيجية المرتبطة بهذه الأهداف:**\n`;
            for (const kpi of uniqueKpis) {
              initiativeCompletionText += `- **${kpi.id}:** ${kpi.name}\n`;
            }
            initiativeCompletionText += "\n";
          }
        }
      }
      
      if (allGapDomains && Array.isArray(allGapDomains)) {
        let bridgedGapsText = "";
        for (const domain of allGapDomains) {
          if (domain.gaps && Array.isArray(domain.gaps)) {
            for (const gap of domain.gaps) {
              if (gap.bridgingInitiative && typeof gap.bridgingInitiative === 'string' &&
                  (gap.bridgingInitiative === initiative.name || gap.bridgingInitiative.includes(initiative.name))) {
                if (bridgedGapsText === "") {
                  bridgedGapsText += `**الفجوات التي تساهم هذه المبادرة في معالجتها:**\n\n`;
                }
                bridgedGapsText += `#### فجوة: ${gap.description}\n`;
                if (gap.impact) {
                  bridgedGapsText += `- **التأثير السلبي للفجوة (قبل المعالجة):** ${gap.impact}\n`;
                }
                if (gap.futureState) {
                  bridgedGapsText += `- **الوضع المستقبلي المستهدف (بعد المعالجة):** ${gap.futureState}\n`;
                }
                bridgedGapsText += "\n";
              }
            }
          }
        }
        if (bridgedGapsText) {
           initiativeCompletionText += `### تأثير المبادرة (من خلال معالجة الفجوات):\n${bridgedGapsText}`;
        }
      }

      const initiativeSpecificPrompts: string[] = [
        `ما هي تفاصيل مبادرة "${initiative.name}"؟`, `معلومات عن مبادرة "${initiative.name}"`,
        `حدثني عن مبادرة "${initiative.name}"`, `ما هي مبادرة "${initiative.name}" (المعرف ${initiative.id})؟`,
        `تفاصيل المبادرة ${initiative.id}`, `مبادرة ${initiative.id}`, `"${initiative.name}"`,
        `ما هي أهداف مبادرة "${initiative.name}"؟`, 
        `ما هو تأثير مبادرة "${initiative.name}"؟`,
        `ما هو الأثر من تنفيذ مبادرة "${initiative.name}"؟`, `ماذا تحقق مبادرة "${initiative.name}"؟`,
        `ما هي الأهداف الاستراتيجية المرتبطة بمبادرة "${initiative.name}"؟`,
        `اذكر الأهداف الاستراتيجية لمبادرة "${initiative.name}"`,
        `ما هي المؤشرات الاستراتيجية المرتبطة بمبادرة "${initiative.name}"؟`,
        `ما هي مؤشرات الأداء المتأثرة بمبادرة "${initiative.name}"؟`,
        `اذكر المؤشرات الاستراتيجية المرتبطة والمتأثرة بمبادرة "${initiative.name}"`,
        `ما هي المشاريع التابعة لمبادرة "${initiative.name}"؟`,
        `ما هي الفجوات التي تعالجها مبادرة "${initiative.name}"؟`
      ];
      
      const initiativeNameParts = initiative.name.split(/[\s-]+/);
      if (initiativeNameParts.length > 2 && initiative.name.startsWith("مبادرة")) { 
          const shortName = initiativeNameParts.slice(1).join(" ");
          if (shortName.length > 3) {
            initiativeSpecificPrompts.push(`ما هي مبادرة "${shortName}"؟`);
            initiativeSpecificPrompts.push(`تفاصيل مبادرة "${shortName}"`);
            initiativeSpecificPrompts.push(`الأهداف الاستراتيجية لمبادرة "${shortName}"`);
          }
      }
      
      Array.from(new Set(initiativeSpecificPrompts)).forEach(promptText => {
        const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
        uniqueEntriesMap.set(normalizedKey, {
          prompt: promptText, completion: initiativeCompletionText,
          sourcePath: `digitalTransformationStrategy.digitalTransformationInitiatives.initiativeDetails.${initiative.id}`
        });
      });
    }
  }
  
  // Step 8: Add/Overwrite with Detailed Strategic Objective-Specific Entries (Vision 2030 Alignment)
  if (allStrategicObjectives && Array.isArray(allStrategicObjectives)) {
    for (const objective of allStrategicObjectives) {
      if (!objective.id || !objective.name) continue;

      // Check for vision2030Alignment field (must be added to metadata.json for each objective)
      if (objective.vision2030Alignment && Array.isArray(objective.vision2030Alignment) && objective.vision2030Alignment.length > 0) {
        let objectiveVisionCompletionText = `## مساهمة هدف التحول الرقمي: "${objective.name}" (المعرف: ${objective.id}) في رؤية المملكة 2030\n\n`;
        objectiveVisionCompletionText += `يساهم هدف التحول الرقمي **"${objective.name}"** في تحقيق أهداف رؤية المملكة 2030 من خلال النقاط التالية:\n`;
        objective.vision2030Alignment.forEach((alignment: any) => {
          if (alignment.visionObjective && alignment.projectContribution) { // Re-using projectContribution for general contribution description
            objectiveVisionCompletionText += `- **${alignment.visionObjective.trim()}:** ${alignment.projectContribution.trim()}\n`;
          }
        });
        objectiveVisionCompletionText += "\n";

        const objectiveVisionPrompts: string[] = [
          `كيف يساهم هدف ${objective.name} في تحقيق رؤية المملكة 2030م؟`,
          `كيف يساهم هدف ${objective.name} في تحقيق رؤية 2030؟`,
          `مواءمة هدف ${objective.name} مع رؤية 2030`,
          `مساهمة هدف ${objective.name} في رؤية 2030`,
          `ما هي مساهمة هدف ${objective.name} في رؤية المملكة 2030؟`,
          `كيف يدعم هدف ${objective.name} رؤية المملكة 2030؟`
        ];

        Array.from(new Set(objectiveVisionPrompts)).forEach(promptText => {
          const normalizedKey = promptText.trim().toLowerCase().replace(/[؟.,!]/g, '');
          uniqueEntriesMap.set(normalizedKey, {
            prompt: promptText,
            completion: objectiveVisionCompletionText,
            sourcePath: `digitalTransformationStrategy.strategicHouse.objectivesData.objectives.${objective.id}.vision2030Alignment`
          });
        });
      }
    }
  }


  const finalEntries = Array.from(uniqueEntriesMap.values());
  return finalEntries;
}


async function loadKnowledgeBase(): Promise<void> {
  try {
    const text = await fetchFileContent('./data_dt_v03.json'); // CRITICAL FIX: Load the correct data file
    if (!text.trim()) {
        knowledgeBase = [];
        return;
    }
    
    let parsedData;
    try {
        parsedData = JSON.parse(text);
    } catch (parseError) {
        console.error("[loadKnowledgeBase] Failed to parse knowledge base (data_dt_v03.json) as JSON:", parseError);
        displayMessage("حدث خطأ أثناء تحليل البيانات المصدرية. قد لا أتمكن من الإجابة على الأسئلة بشكل صحيح.", 'error');
        knowledgeBase = []; 
        return;
    }

    if (parsedData && typeof parsedData === 'object' && parsedData !== null) {
        knowledgeBase = await transformMetadataToKnowledgeBase(parsedData); 
        
    } else {
        knowledgeBase = [];
        displayMessage("البيانات المصدرية ليست بالتنسيق المتوقع. لا يمكن تحميل قاعدة المعرفة.", 'error');
    }

  } catch (error) {
    console.error("[loadKnowledgeBase] Failed to load or process knowledge base (data_dt_v03.json):", error);
    displayMessage("حدث خطأ أثناء تحميل قاعدة البيانات المعرفية. قد لا أتمكن من الإجابة على الأسئلة بشكل صحيح.", 'error');
    knowledgeBase = []; 
  }
}


// ===== دالة البحث المطورة =====
function getBestMatch(userInput: string, knowledge: KnowledgeEntry[]): KnowledgeEntry[] {
  const normalizedUserInput = userInput.trim().toLowerCase().replace(/[؟.,!]/g, '');

  if (!normalizedUserInput) {
    return [];
  }

  for (const entry of knowledge) {
    if (!entry.prompt || typeof entry.prompt !== 'string' || !entry.completion) continue;
    const normalizedKbPrompt = entry.prompt.trim().toLowerCase().replace(/[؟.,!]/g, '');
    if (normalizedKbPrompt === normalizedUserInput) {
      return [entry];
    }
  }

  const arabicStopWords = new Set([
      'في', 'على', 'الى', 'إلى', 'عن', 'و', 'أو', 'ثم', 'ما', 'هي', 'ماهي',
      'هو', 'هل', 'يا', 'اي', 'أي', 'ان', 'أن', 'اذا', 'إذا', 'لكن',
      'قد', 'تم', 'مع', 'كذلك', 'مثل', 'هذا', 'هذه', 'ذلك', 'تلك', 'به',
      'فيه', 'عليه', 'إليه', 'عنه', 'لي', 'له', 'لها', 'لهم', 'لنا',
      'جدا', 'ايضا', 'أيضاً', 'فقط', 'بعض', 'كل', 'جميع', 'اذكر', 'ماذا', 'كيف',
      'بشكل', 'عام', 'حول', 'بخصوص', 'عن ماذا', 'تكلم عن', 'اشرح', 'وضح',
      'الخاصة', 'المتعلقة', 'ضمن', 'قسم' 
  ]);

  const userInputKeywords = normalizedUserInput.split(/\s+/).filter(w => w.length > 1 && !arabicStopWords.has(w));

  if (userInputKeywords.length === 0) {
     const scoredMatchesFallback: { entry: KnowledgeEntry; score: number }[] = [];
     for (const entry of knowledge) {
        if (!entry.prompt || typeof entry.prompt !== 'string' || !entry.completion) continue;
        const kbPromptText = entry.prompt.trim().toLowerCase().replace(/[؟.,!]/g, '');
        if (kbPromptText.includes(normalizedUserInput)) { 
            scoredMatchesFallback.push({ entry, score: normalizedUserInput.length / kbPromptText.length });
        }
     }
     if (scoredMatchesFallback.length > 0) {
        scoredMatchesFallback.sort((a, b) => b.score - a.score);
        return [scoredMatchesFallback[0].entry];
     }
    return [];
  }

  const scoredMatches: { entry: KnowledgeEntry; score: number }[] = [];
  for (const entry of knowledge) {
    if (!entry.prompt || typeof entry.prompt !== 'string' || !entry.completion) continue;
    
    const kbPromptText = entry.prompt.trim().toLowerCase().replace(/[؟.,!]/g, '');
    const kbPromptKeywords = new Set(kbPromptText.split(/\s+/).filter(w => w.length > 1 && !arabicStopWords.has(w)));

    let currentMatchedWordCount = 0;
    let keywordMatchScore = 0; 
    
    for (const userKeyword of userInputKeywords) {
      let keywordFoundInPrompt = false;
      for (const kbKeyword of kbPromptKeywords) { 
        if (kbKeyword.includes(userKeyword) || userKeyword.includes(kbKeyword)) { 
          currentMatchedWordCount++;
          keywordMatchScore += Math.max(userKeyword.length, kbKeyword.length); 
          keywordFoundInPrompt = true;
          break; 
        }
      }
      if (!keywordFoundInPrompt && kbPromptText.includes(userKeyword)){
        currentMatchedWordCount += 0.5; 
        keywordMatchScore += userKeyword.length * 0.5;
      }
    }

    if (currentMatchedWordCount > 0) {
      const totalPossibleKeywordScore = userInputKeywords.join("").length; 
      const relevanceScore = totalPossibleKeywordScore > 0 ? keywordMatchScore / totalPossibleKeywordScore : 0;
      const densityScore = userInputKeywords.length > 0 ? currentMatchedWordCount / userInputKeywords.length : 0;
      
      let finalScore = (relevanceScore + densityScore) / 2;

      if (densityScore === 1 && userInputKeywords.length > 0) { 
        finalScore += 0.1; 
        if (kbPromptKeywords.size <= userInputKeywords.length + 2 && kbPromptKeywords.size >= userInputKeywords.length -2 ) { 
          finalScore += 0.1; 
        }
      }
      const lengthRatio = Math.min(userInputKeywords.join(" ").length, kbPromptText.length) / Math.max(userInputKeywords.join(" ").length, kbPromptText.length);
      finalScore += (lengthRatio * 0.1);


      finalScore = Math.min(finalScore, 1.0); 

      scoredMatches.push({
        entry: entry,
        score: finalScore,
      });
    }
  }

  if (scoredMatches.length === 0) {
    return [];
  }
  
  scoredMatches.sort((a, b) => b.score - a.score);
  
  const bestScore = scoredMatches[0].score;
  const relevanceThreshold = Math.max(0.25, bestScore * 0.60); 

  const filteredMatches = scoredMatches.filter(m => m.score >= relevanceThreshold); 
  
  if(filteredMatches.length === 0 && scoredMatches.length > 0 && scoredMatches[0].score > 0.1) { 
      return [scoredMatches[0].entry];
  } else if (filteredMatches.length === 0) {
      return [];
  }

  const topMatches = filteredMatches.slice(0, MAX_CANDIDATES_TO_SEND).map(m => m.entry);
  return topMatches;
}


const initialHistory: Content[] = [
  {
    role: "user",
    parts: [{ text: "مرحباً" }],
  },
  {
    role: "model",
    parts: [{ text: "أهلاً بك! أنا مساعدك المتخصص في استراتيجية التحول الرقمي لهيئة الهلال الأحمر السعودي. كيف يمكنني خدمتك اليوم؟" }],
  }
];

const chat: Chat = ai.chats.create({
  model: modelName,
  history: initialHistory,
  config: {
    safetySettings: safetySettings,
    systemInstruction: systemInstructionTextRAG,
  }
});


const chatInput = document.getElementById('message-input') as HTMLInputElement; // Corrected type to HTMLInputElement
const sendButton = document.getElementById('send-button') as HTMLButtonElement;
const chatOutput = document.getElementById('chat-messages') as HTMLDivElement;
const loadingIndicator = document.getElementById('loading-indicator') as HTMLDivElement;
// errorMessageGlobalDiv is already defined at the top

function displayMessage(message: string, sender: 'user' | 'model' | 'error', sources?: { uri: string; title?: string }[]) {
  const messageDiv = document.createElement('div');
  messageDiv.classList.add('chat-message'); 

  if (sender === 'user') {
    messageDiv.classList.add('user-message');
  } else if (sender === 'model') {
    messageDiv.classList.add('bot-message');
  } else if (sender === 'error') {
    messageDiv.classList.add('error-message'); // Class for styling error messages in chat
  }

  messageDiv.setAttribute('role', 'article');
  if (sender === 'user') {
    messageDiv.setAttribute('aria-label', 'رسالة المستخدم');
  } else if (sender === 'model') {
    messageDiv.setAttribute('aria-label', 'رسالة النموذج');
  } else {
    messageDiv.setAttribute('aria-label', 'رسالة خطأ');
  }

  const contentDiv = document.createElement('div');
  contentDiv.classList.add('message-content');

  if (sender === 'model' || sender === 'error') {
    const rawHtml = marked.parse(message); 
    const sanitizedHtml = typeof rawHtml === 'string' ? rawHtml.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '') : '';
    contentDiv.innerHTML = sanitizedHtml;
  } else {
    contentDiv.textContent = message;
  }
  messageDiv.appendChild(contentDiv);

  if (sender === 'model' && sources && sources.length > 0) {
    // This block is intentionally kept but displayGroundingSources is not called
    // to prevent displaying the explicit list of URLs as per user request.
  }

  if (chatOutput) { 
      chatOutput.appendChild(messageDiv);
      chatOutput.scrollTop = chatOutput.scrollHeight;
  } else {
      console.error("Chat output area not found!");
  }
}

function displayGroundingSources(groundingChunks: any[]) {
    // This function is defined but its call is commented out in sendMessage
    // to prevent displaying the explicit list of URLs as per user request.
    if (!groundingChunks || groundingChunks.length === 0 || !chatOutput) return;

    const sources: { uri: string; title?: string }[] = [];
    groundingChunks.forEach(chunk => {
        if (chunk.web && chunk.web.uri) {
            sources.push({ uri: chunk.web.uri, title: chunk.web.title || chunk.web.uri });
        }
    });

    if (sources.length > 0) {
        const lastModelMessage = chatOutput.querySelector('.bot-message:last-child');
        if (lastModelMessage) {
            let sourcesContainer = lastModelMessage.querySelector('.message-sources') as HTMLDivElement;
            if (!sourcesContainer) {
                sourcesContainer = document.createElement('div');
                sourcesContainer.classList.add('message-sources');
                const sourcesTitle = document.createElement('h5');
                sourcesTitle.textContent = 'المصادر المستند إليها من البحث:';
                sourcesContainer.appendChild(sourcesTitle);
                const sourcesList = document.createElement('ul');
                sourcesContainer.appendChild(sourcesList);
                lastModelMessage.appendChild(sourcesContainer);
            }
            
            const sourcesList = sourcesContainer.querySelector('ul');
            if (sourcesList) {
                 sources.forEach(source => {
                    const listItem = document.createElement('li');
                    const link = document.createElement('a');
                    link.href = source.uri;
                    link.textContent = source.title || source.uri;
                    link.target = '_blank';
                    link.rel = 'noopener noreferrer';
                    listItem.appendChild(link);
                    sourcesList.appendChild(listItem);
                });
            }
        }
    }
}


async function sendMessage(currentInput: string) {
  userInput = currentInput; 
  if (!userInput.trim()) return;

  displayMessage(userInput, 'user');
  if (chatInput) chatInput.value = ''; 
  if (sendButton) sendButton.disabled = true; 
  if (loadingIndicator) loadingIndicator.style.display = 'block';
  if (errorMessageGlobalDiv) errorMessageGlobalDiv.style.display = 'none';


  let entityNameToUseForSearch: string | null = null;
  let entityTypeForSearch: 'مشروع' | 'هدف' | null = null;
  let useGoogleSearch = false;
  let localContextForVision2030 = "";

  const projectVisionQueryMatch = userInput.match(visionQueryPattern);
  const objectiveVisionQueryMatch = userInput.match(objectiveVisionQueryPattern);
  
  if (projectVisionQueryMatch && projectVisionQueryMatch[2]) { 
    entityNameToUseForSearch = projectVisionQueryMatch[2].trim().replace(/[؟.,!]/g, '');
    entityTypeForSearch = 'مشروع';
    const specificVisionPrompt = `كيف تساهم مستهدفات مشروع ${entityNameToUseForSearch} في تحقيق رؤية المملكة 2030م؟`;
    const specificMatches = getBestMatch(specificVisionPrompt, knowledgeBase);
    
    if (specificMatches.length > 0 && specificMatches[0].completion.includes("المساهمة في تحقيق رؤية المملكة 2030")) {
      const visionSectionMatch = specificMatches[0].completion.match(/### المساهمة في تحقيق رؤية المملكة 2030:([\s\S]*)/);
      if (visionSectionMatch && visionSectionMatch[1] && visionSectionMatch[1].trim().length > 30) {
        localContextForVision2030 = specificMatches[0].completion;
      }
    }
    if (!localContextForVision2030) {
      useGoogleSearch = true;
    }
  } else if (objectiveVisionQueryMatch && objectiveVisionQueryMatch[1]) {
    entityNameToUseForSearch = objectiveVisionQueryMatch[1].trim().replace(/[؟.,!]/g, '');
    entityTypeForSearch = 'هدف';
    const specificVisionPrompt = `كيف يساهم هدف ${entityNameToUseForSearch} في تحقيق رؤية المملكة 2030م؟`;
    const specificMatches = getBestMatch(specificVisionPrompt, knowledgeBase);

    if (specificMatches.length > 0 && specificMatches[0].completion.includes("مساهمة هدف التحول الرقمي") && specificMatches[0].completion.includes("في رؤية المملكة 2030")) {
         const visionSectionMatch = specificMatches[0].completion.match(/## مساهمة هدف التحول الرقمي:.*?في رؤية المملكة 2030([\s\S]*)/);
         if (visionSectionMatch && visionSectionMatch[1] && visionSectionMatch[1].trim().length > 30){
            localContextForVision2030 = specificMatches[0].completion;
         }
    }
     if (!localContextForVision2030) {
      useGoogleSearch = true;
    }
  }


  if (useGoogleSearch && entityNameToUseForSearch && entityTypeForSearch) {
    const searchQueryForGoogle = `مساهمة ${entityTypeForSearch} "${entityNameToUseForSearch}" التابع لهيئة الهلال الأحمر السعودي في تحقيق رؤية المملكة 2030`;
    const systemInstructionForGoogleSearch = `أنت مساعد خبير. مهمتك هي الإجابة على السؤال حول كيف يساهم ${entityTypeForSearch} "${entityNameToUseForSearch}" في تحقيق أهداف رؤية المملكة 2030، بناءً على نتائج البحث المقدمة. قدم إجابة مباشرة ومركزة وصغها كأنها معرفتك الخاصة، **وتجنب تمامًا أي إشارة إلى أنك تبحث أو أن المعلومات من مصادر خارجية أو مواقع ويب.** اشرح المساهمات بوضوح. إذا كانت المساهمات متعددة، استخدم قائمة نقطية لزيادة الوضوح.`;
    
    let responseText = "";
    let currentModelMessageDiv: HTMLDivElement | null = null;
    let currentModelContentDiv: HTMLDivElement | null = null;
    let groundingMetadataFromStream: any[] = [];

    try {
      const streamResult = await ai.models.generateContentStream({
        model: modelName,
        contents: [{ role: "user", parts: [{ text: searchQueryForGoogle }] }],
        config: {
          safetySettings: safetySettings,
          systemInstruction: systemInstructionForGoogleSearch,
          tools: [{ googleSearch: {} }],
        }
      });

      for await (const chunk of streamResult) { 
        const chunkText = chunk.text;
        if (chunkText) {
          responseText += chunkText;
          if (!currentModelMessageDiv) {
            currentModelMessageDiv = document.createElement('div');
            currentModelMessageDiv.classList.add('chat-message', 'bot-message');
            currentModelMessageDiv.setAttribute('role', 'article');
            currentModelMessageDiv.setAttribute('aria-label', 'رسالة النموذج');
            currentModelContentDiv = document.createElement('div');
            currentModelContentDiv.classList.add('message-content');
            currentModelMessageDiv.appendChild(currentModelContentDiv);
            if (chatOutput) chatOutput.appendChild(currentModelMessageDiv);
          }
          if (currentModelContentDiv) {
            const rawHtml = marked.parse(responseText);
            const sanitizedHtml = typeof rawHtml === 'string' ? rawHtml.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '') : '';
            currentModelContentDiv.innerHTML = sanitizedHtml;
          }
        }
        if (chunk.candidates && chunk.candidates[0].groundingMetadata && chunk.candidates[0].groundingMetadata.groundingChunks) {
            const newChunks = chunk.candidates[0].groundingMetadata.groundingChunks;
            newChunks.forEach((nc: any) => {
                if (!groundingMetadataFromStream.some(existingChunk => existingChunk.web && nc.web && existingChunk.web.uri === nc.web.uri)) {
                    groundingMetadataFromStream.push(nc);
                }
            });
        }
        if (chatOutput) chatOutput.scrollTop = chatOutput.scrollHeight;
      }
      
      if (!responseText.trim() && currentModelMessageDiv && currentModelContentDiv) {
        currentModelContentDiv.innerHTML = marked.parse("لم أتمكن من إيجاد إجابة محددة عبر البحث.") as string;
      } else if (!responseText.trim() && !currentModelMessageDiv) {
        displayMessage("لم أتمكن من إيجاد إجابة محددة عبر البحث.", 'model');
      }

      /* 
      // تم تعطيل عرض مصادر البحث بناءً على طلب المستخدم
      if (groundingMetadataFromStream.length > 0) {
        // displayGroundingSources(groundingMetadataFromStream); 
      }
      */

    } catch (error: any) {
      console.error(`Error sending message to Gemini with Google Search (for ${entityTypeForSearch}):`, error);
      displayMessage("عذرًا، حدث خطأ أثناء محاولة البحث عن إجابة.", 'error');
    } finally {
      if (loadingIndicator) loadingIndicator.style.display = 'none';
      if (sendButton) sendButton.disabled = false;
      if (chatInput) chatInput.focus();
    }
  } else { 
    const bestMatches = getBestMatch(userInput, knowledgeBase);
    let contextToUse = localContextForVision2030; 

    if (!contextToUse) { 
        if (!bestMatches || bestMatches.length === 0) {
            const promptWithoutContext = `
                **السؤال من المستخدم:**
                ${userInput}
                **التعليمات:**
                أنت مساعد خبير متخصص في استراتيجية التحول الرقمي لهيئة الهلال الأحمر السعودي.
                أجب على "السؤال من المستخدم" أعلاه بناءً على فهمك العام لموضوع استراتيجيات التحول الرقمي.
                إذا كان السؤال يتطلب معلومات محددة جدًا لا تملكها كجزء من خبرتك العامة، يمكنك توضيح أنك لا تملك التفاصيل المطلوبة للإجابة على هذا الجانب المحدد، دون الإشارة إلى بحث في وثائق أو مصادر.
            `;
            try {
                const resultStream = await chat.sendMessageStream({ message: promptWithoutContext });
                let responseText = "";
                let currentModelMessageDiv: HTMLDivElement | null = null;
                let currentModelContentDiv: HTMLDivElement | null = null;

                for await (const chunk of resultStream) { 
                    const chunkText = chunk.text; 
                    if (chunkText) {
                        responseText += chunkText;
                        if (!currentModelMessageDiv) {
                            currentModelMessageDiv = document.createElement('div');
                            currentModelMessageDiv.classList.add('chat-message', 'bot-message');
                            currentModelContentDiv = document.createElement('div');
                            currentModelContentDiv.classList.add('message-content');
                            currentModelMessageDiv.appendChild(currentModelContentDiv);
                            if (chatOutput) chatOutput.appendChild(currentModelMessageDiv);
                        }
                        if (currentModelContentDiv) {
                            currentModelContentDiv.innerHTML = marked.parse(responseText) as string;
                        }
                    }
                    if (chatOutput) chatOutput.scrollTop = chatOutput.scrollHeight;
                }
                if (!responseText.trim() && currentModelMessageDiv && currentModelContentDiv) { 
                    currentModelContentDiv.innerHTML = marked.parse("لم أتمكن من إيجاد إجابة محددة بناءً على المعلومات المتوفرة.") as string;
                } else if (!responseText.trim() && !currentModelMessageDiv) { 
                     displayMessage("لم أتمكن من إيجاد إجابة محددة بناءً على المعلومات المتوفرة.", 'model');
                }

            } catch (error: any) {
                console.error("Error sending message to Gemini (no context):", error);
                displayMessage("عذرًا، حدث خطأ أثناء معالجة طلبك.", 'error');
            } finally {
                if (loadingIndicator) loadingIndicator.style.display = 'none';
                if (sendButton) sendButton.disabled = false;
                if (chatInput) chatInput.focus();
            }
            return;
        }
        contextToUse = bestMatches[0].completion;
    }
    
    const promptForModel = `
      أنت خبير باستراتيجية التحول الرقمي لهيئة الهلال الأحمر السعودي.
      استخدم المعلومات التالية **فقط وحصرياً** للإجابة على السؤال:
      ---
      ${contextToUse}
      ---
      **السؤال من المستخدم:**
      ${userInput}

      **تعليمات صارمة للإجابة:**
      1.  أجب على "السؤال من المستخدم" بدقة متناهية، مستنداً **فقط** إلى "المعلومات" أعلاه.
      2.  حلل "المعلومات" بعناية لاستخلاص الإجابة. إذا كانت "المعلومات" هي نفسها الإجابة المطلوبة (مثلاً نص مُلخص جاهز)، قم بتقديمها بأسلوب طبيعي.
      3.  صغ إجابتك بأسلوب طبيعي، واضح، وموجز، كأن هذه هي معرفتك المباشرة.
      4.  **تحذير حاسم: لا تذكر إطلاقاً، تحت أي ظرف، كلمات مثل "المعلومات المقدمة"، "السياق"، "المصدر"، "الوثيقة"، "الملف"، "JSON"، أو أي إشارة إلى كيفية حصولك على المعلومة. أجب كخبير مباشر.**
      5.  إذا كانت "المعلومات" لا تحتوي على إجابة كافية للسؤال (حتى بعد التحليل)، أجب بوضوح: "لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال المحدد حاليًا بناءً على ما لدي." (استخدم هذه الصياغة تحديداً).
      6.  إذا كانت "المعلومات" تحتوي تفاصيل يمكن عرضها بشكل أفضل كقائمة (نقطية أو مرقمة)، استخدم تنسيق القوائم لتعزيز الوضوح، ما لم تكن "المعلومات" نفسها مُنسقة بالفعل كقائمة مناسبة.
      7.  **توضيح هام للسنوات:** عند الإشارة إلى 'السنة الاولى'، فإنها تعني عام 2026. 'السنة الثانية' تعني عام 2027. و'السنة الثالثة' تعني عام 2028. استخدم هذه المعلومة عند تحليل 'المعلومات' المقدمة إذا كانت تحتوي على هذه المصطلحات.
      `;

    try {
      const resultStream = await chat.sendMessageStream({ message: promptForModel });
      let responseText = "";
      let currentModelMessageDiv: HTMLDivElement | null = null;
      let currentModelContentDiv: HTMLDivElement | null = null;

      for await (const chunk of resultStream) { 
        const chunkText = chunk.text; 
        if (chunkText) {
          responseText += chunkText;
          if (!currentModelMessageDiv) {
            currentModelMessageDiv = document.createElement('div');
            currentModelMessageDiv.classList.add('chat-message', 'bot-message');
            currentModelContentDiv = document.createElement('div');
            currentModelContentDiv.classList.add('message-content');
            currentModelMessageDiv.appendChild(currentModelContentDiv);
            if (chatOutput) chatOutput.appendChild(currentModelMessageDiv);
          }
          if (currentModelContentDiv) {
              currentModelContentDiv.innerHTML = marked.parse(responseText) as string;
          }
        }
        if (chatOutput) chatOutput.scrollTop = chatOutput.scrollHeight;
      }
      if (!responseText.trim() && currentModelMessageDiv && currentModelContentDiv) { 
          currentModelContentDiv.innerHTML = marked.parse("لم أتمكن من إيجاد إجابة محددة بناءً على المعلومات المتوفرة.") as string;
      } else if (!responseText.trim() && !currentModelMessageDiv) { 
           displayMessage("لم أتمكن من إيجاد إجابة محددة بناءً على المعلومات المتوفرة.", 'model');
      }

    } catch (error: any) {
      console.error("Error sending message to Gemini:", error);
      let displayError = "عذرًا، حدث خطأ أثناء معالجة طلبك.";
      if (error.message) {
          if (error.message.includes('DEADLINE_EXCEEDED')) {
              displayError = "انتهت مهلة الطلب. الرجاء المحاولة مرة أخرى.";
          } else if (error.message.includes('API_KEY_INVALID') || error.message.includes('API key not valid') || (error.toString && error.toString().includes('API key not valid'))) {
              displayError = "مفتاح API غير صالح أو غير مصرح به. يرجى مراجعة الإعدادات.";
          } else if (error.message.toLowerCase().includes('quota') || (error.status === 429 || (error.error && error.error.code === 429)) ) {
              displayError = "تم تجاوز حد الطلبات. يرجى المحاولة لاحقًا.";
          } else if (error.message.includes('SAFETY') || (error.error && error.error.message && error.error.message.includes('SAFETY'))) {
              displayError = "تم حظر الرد بسبب إعدادات السلامة.";
          } else if (error.message.includes('fetch') && error.message.toLowerCase().includes('failed')) {
              displayError = "حدث خطأ في الاتصال بالشبكة. يرجى التحقق من اتصالك بالإنترنت والمحاولة مرة أخرى.";
          }
      }
      displayMessage(displayError, 'error');
      if (errorMessageGlobalDiv) {
          errorMessageGlobalDiv.textContent = displayError;
          errorMessageGlobalDiv.style.display = 'block';
      }
    } finally {
      if (loadingIndicator) loadingIndicator.style.display = 'none';
      if (sendButton) sendButton.disabled = false;
      if (chatInput) chatInput.focus();
    }
  }
}

if (chatInput) { 
    chatInput.addEventListener('keypress', (event) => {
      if (event.key === 'Enter' && !event.shiftKey) { // !event.shiftKey for single line input behavior
        event.preventDefault();
        sendMessage(chatInput.value);
      }
    });
}


if (sendButton) { 
    sendButton.addEventListener('click', () => {
      if (chatInput) sendMessage(chatInput.value); 
    });
}


setTimeout(() => {
    if (initialHistory.length > 1 && initialHistory[initialHistory.length -1].role === 'model'){
        const lastModelMessage = initialHistory[initialHistory.length -1].parts[0].text;
        if(lastModelMessage && chatOutput) { 
            const existingMessages = chatOutput.querySelectorAll('.bot-message .message-content'); 
            let alreadyDisplayed = false;
            const parsedLastModelMessage = marked.parse(lastModelMessage.trim()) as string;
            existingMessages.forEach(msgElement => {
                if (msgElement.innerHTML?.trim() === parsedLastModelMessage.trim()) {
                    alreadyDisplayed = true;
                }
            });
            if (!alreadyDisplayed) {
                displayMessage(lastModelMessage, 'model');
            }
        }
    }
}, 0);


window.addEventListener('error', (event) => {
  if (event.message.includes("API_KEY not set") || event.message.includes("API_KEY is not configured")) {
    // errorMessageGlobalDiv is already defined and used at the top for initial API_KEY check
    if (errorMessageGlobalDiv) {
        errorMessageGlobalDiv.textContent = "API_KEY is not configured. Please check the application setup and ensure the API_KEY environment variable is correctly set.";
        errorMessageGlobalDiv.style.display = 'block';
    }
      if (sendButton) sendButton.disabled = true;
      if (chatInput) chatInput.disabled = true;
      if (suggestedQuestionsContainer) suggestedQuestionsContainer.style.display = 'none'; 
  }
});

const suggestedQuestions = [
    "ما هي رؤية التحول الرقمي؟",
    "اذكر الركائز الاستراتيجية للتحول الرقمي.",
    "ما هي مشاريع السنة الاولى (2026)؟"
];

function displaySuggestedQuestions() {
    const area = document.getElementById('suggested-questions-area');
    if (!area || !suggestedQuestionsContainer) return;

    area.innerHTML = ''; // Clear previous questions if any

    if (API_KEY) { // Only display if API_KEY is available
        suggestedQuestions.forEach(questionText => {
            const button = document.createElement('button');
            button.textContent = questionText;
            button.classList.add('suggested-question-button');
            button.setAttribute('role', 'button');
            button.setAttribute('aria-label', `اطرح السؤال: ${questionText}`);
            button.addEventListener('click', () => {
                if (chatInput) {
                    chatInput.value = questionText;
                    sendMessage(questionText);
                    chatInput.focus();
                }
            });
            area.appendChild(button);
        });
        suggestedQuestionsContainer.style.display = 'block';
    } else {
        suggestedQuestionsContainer.style.display = 'none';
    }
}


(async () => {
  await loadKnowledgeBase();
  displaySuggestedQuestions(); // Display suggested questions after loading knowledge base (or attempting to)
  if (chatInput && API_KEY) { // Only focus if API_KEY is set
      chatInput.focus();
  }
})();
