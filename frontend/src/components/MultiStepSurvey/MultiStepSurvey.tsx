import { forwardRef, useState } from "react";
import { useNavigate } from "react-router";
import { ScaleLoader } from "react-spinners";
import classNames from "classnames";

import { extractAnswer } from "./utils";

import LogoContainer from "@components/LogoContainer/LogoContainer";
import Button from "@components/Button/Button";

import RadioButton from "./components/RadioButton/RadioButton";
import ProgressNav from "./components/ProgressNav/ProgressNav";
import DefaultOutro from "./components/DefaultOutro/DefaultOutro";
import DefaultIntro from "./components/DefaultIntro/DefaultIntro";

import { MultiStepSurveyProps } from "./MultiStepSurvey.props";
import { SurveyData } from "./MultiStepSurvey.types";

import classes from "./MultiStepSurvey.module.css";

const postSurveyResults = (surveyData: SurveyData, userMeta?: any) => {
  const result = {
    ...(userMeta ? { userMeta } : {}),
    answers: surveyData,
  };

  alert(JSON.stringify(result, null, 2));
};

const MultiStepSurvey = forwardRef<HTMLDivElement, MultiStepSurveyProps>(
  (
    {
      questions,
      Intro = DefaultIntro,
      Outro = DefaultOutro,
      userMeta = null,
      onComplete = postSurveyResults,
      loading = false,
      onExit,
      onLogoClick,
    },
    ref
  ) => {
    const [questionIDs, setQuestionIDs] = useState<string[]>(
      Object.keys(questions)
        .filter((key) => !key.includes("."))
        .sort((a, b) => parseInt(a) - parseInt(b))
    );
    const [surveyData, setSurveyData] = useState<SurveyData>(
      Object.fromEntries(questionIDs.map((id) => [id, null]))
    );
    const [currentStep, setCurrentStep] = useState<number>(-1);

    const navigate = useNavigate();

    const isLastStep = () => {
      return currentStep === questionIDs.length - 1;
    };

    const AreAllQuestionsAnswered = () => {
      return Object.values(surveyData).every((answer) => answer !== null);
    };

    if (loading) {
      return <ScaleLoader color={"var(--solitude-100)"} ref={ref} />;
    }

    if (currentStep <= -1 && Intro) {
      return (
        <Intro
          onStepChange={setCurrentStep}
          onLogoClick={onLogoClick}
          ref={ref}
        />
      );
    } else if (currentStep >= questionIDs.length && Outro) {
      return (
        <Outro
          onExit={onExit || (() => navigate("/login"))}
          onLogoClick={onLogoClick}
          ref={ref}
        />
      );
    } else {
      const currentQuestionId = questionIDs[currentStep];
      const currentQuestion = questions[currentQuestionId];

      const backButtonClasses = classNames(classes.MultiStepSurveyButton, {
        [classes.HiddenButton]: currentStep <= 0,
      });
      const nextButtonClasses = classNames(classes.MultiStepSurveyButton, {
        [classes.HiddenButton]: isLastStep()
          ? !AreAllQuestionsAnswered()
          : !surveyData[questionIDs[currentStep]],
      });

      const handleRadioClick = (answerIndex: number) => {
        const answer = currentQuestion.answers[answerIndex];
        const subquestionPrefix = `${Number(currentQuestionId)}.`;

        if (typeof answer === "string") {
          setSurveyData((prev) => {
            const updated = { ...prev };

            Object.keys(updated).forEach((key) => {
              if (key.startsWith(subquestionPrefix)) {
                delete updated[key];
              }
            });

            return {
              ...updated,
              [currentQuestionId]: answerIndex,
            };
          });

          setQuestionIDs((prev) =>
            prev.filter((id) => !id.startsWith(subquestionPrefix))
          );
        } else {
          const nextQuestionId = Object.values(answer)[0];

          const cleanedAnswerIDs = questionIDs.filter(
            (id) => !id.startsWith(subquestionPrefix)
          );
          const updatedAnswerIDs = [
            ...cleanedAnswerIDs.slice(0, currentStep + 1),
            nextQuestionId,
            ...cleanedAnswerIDs.slice(currentStep + 1),
          ];
          setQuestionIDs(updatedAnswerIDs);

          setSurveyData((prev) => {
            const updated = { ...prev };

            Object.keys(updated).forEach((key) => {
              if (key.startsWith(subquestionPrefix)) {
                delete updated[key];
              }
            });

            return {
              ...updated,
              [currentQuestionId]: answerIndex,
              [nextQuestionId]: null,
            };
          });
        }

        if (!isLastStep()) {
          setCurrentStep((prev) => prev + 1);
        }
      };

      return (
        <LogoContainer onLogoClick={onLogoClick}>
          <div className={classes.MultiStepSurvey} ref={ref}>
            <ProgressNav
              step={currentStep}
              questions={questions}
              questionIDs={questionIDs}
              surveyData={surveyData}
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
                    value={index}
                    displayedValue={extractAnswer(answer)}
                    onClick={handleRadioClick}
                    isSelected={surveyData[currentQuestionId] === index}
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
                    onComplete(surveyData, userMeta);
                  }

                  setCurrentStep((prev) => ++prev);
                }}
              >
                {isLastStep() ? "Завершить" : "Далее"}
              </Button>
            </div>
          </div>
        </LogoContainer>
      );
    }
  }
);

export default MultiStepSurvey;
