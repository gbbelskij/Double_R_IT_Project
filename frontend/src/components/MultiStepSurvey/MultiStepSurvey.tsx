import { useState } from "react";
import classNames from "classnames";

import RadioButton from "./components/RadioButton/RadioButton";
import ProgressNav from "./components/ProgressNav/ProgressNav";
import DefaultOutro from "./components/DefaultOutro/DefaultOutro";
import DefaultIntro from "./components/DefaultIntro/DefaultIntro";
import LogoContainer from "@components/LogoContainer/LogoContainer";
import Button from "@components/Button/Button";

import { extractAnswer } from "./utils";

import { MultiStepSurveyProps } from "./MultiStepSurvey.props";
import { AnswerEntries } from "./MultiStepSurvey.types";
import { Answer } from "src/types/question";

import classes from "./MultiStepSurvey.module.css";

const MultiStepSurvey: React.FC<MultiStepSurveyProps> = ({
  questions,
  Intro = DefaultIntro,
  Outro = DefaultOutro,
  userMeta = null,
}) => {
  const [answerIDs, setAnswerIDs] = useState<string[]>(
    Object.keys(questions)
      .filter((key) => !key.includes("."))
      .sort((a, b) => parseInt(a) - parseInt(b))
  );
  const [currentStep, setCurrentStep] = useState<number>(-1);
  const [selectedAnswers, setSelectedAnswers] = useState<AnswerEntries>(
    Object.fromEntries(answerIDs.map((id) => [id, null]))
  );

  const isLastStep = () => {
    return currentStep === answerIDs.length - 1;
  };

  const AreAllQuestionsAnswered = () => {
    return Object.values(selectedAnswers).every((answer) => answer !== null);
  };

  const postSurveyResults = () => {
    const result = {
      ...(userMeta ? { userMeta } : {}),
      answers: selectedAnswers,
    };

    alert(JSON.stringify(result, null, 2));
  };

  if (currentStep <= -1) {
    return <Intro onStepChange={setCurrentStep} />;
  } else if (currentStep >= answerIDs.length) {
    return <Outro />;
  } else {
    const currentQuestion = questions[answerIDs[currentStep]];

    const backButtonClasses = classNames({
      [classes.HiddenButton]: currentStep <= 0,
    });
    const nextButtonClasses = classNames({
      [classes.HiddenButton]: isLastStep()
        ? !AreAllQuestionsAnswered()
        : !selectedAnswers[answerIDs[currentStep]],
    });

    const handleRadioClick = (answer: Answer) => {
      const currentQuestionId = answerIDs[currentStep];

      if (typeof answer === "string") {
        setSelectedAnswers((prev) => ({
          ...prev,
          [currentQuestionId]: answer,
        }));
      } else {
        const nextQuestionId = Object.values(answer)[0];
        const selectedValue = Object.keys(answer)[0];

        const updatedAnswerIDs = [...answerIDs];
        updatedAnswerIDs.splice(currentStep + 1, 0, nextQuestionId);
        setAnswerIDs(updatedAnswerIDs);

        setSelectedAnswers((prev) => ({
          ...prev,
          [currentQuestionId]: selectedValue,
          [nextQuestionId]: null,
        }));
      }

      if (!isLastStep()) {
        setCurrentStep((prev) => prev + 1);
      }
    };

    return (
      <LogoContainer>
        <section className={classes.MultiStepSurvey}>
          <ProgressNav
            questions={questions}
            answerIDs={answerIDs}
            selectedAnswers={selectedAnswers}
            onStepChange={setCurrentStep}
          />

          <div className={classes.MultiStepSurveyChoice}>
            <h2 className={classes.MultiStepSurveyQuestion}>
              {currentQuestion.text}
            </h2>

            <div className={classes.MultiStepSurveyAnswers}>
              {currentQuestion.answers.map((answer, index) => (
                <RadioButton
                  key={index}
                  value={answer}
                  onClick={handleRadioClick}
                  isSelected={
                    selectedAnswers[answerIDs[currentStep]] ===
                    extractAnswer(answer)
                  }
                />
              ))}
            </div>
          </div>

          <div className={classes.MultiStepSurveyButtons}>
            <Button
              className={backButtonClasses}
              size="medium"
              color="inverse"
              isFullWidth
              onClick={() => setCurrentStep((prev) => prev - 1)}
            >
              Назад
            </Button>

            <Button
              className={nextButtonClasses}
              size="medium"
              color="inverse"
              isFullWidth
              onClick={() => {
                if (isLastStep()) {
                  postSurveyResults();
                }

                setCurrentStep((prev) => prev + 1);
              }}
            >
              {isLastStep() ? "Завершить" : "Далее"}
            </Button>
          </div>
        </section>
      </LogoContainer>
    );
  }
};

export default MultiStepSurvey;
