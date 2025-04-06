import { useState } from "react";
import classNames from "classnames";

import RadioButton from "./components/RadioButton/RadioButton";
import ProgressNav from "./components/ProgressNav/ProgressNav";
import DefaultOutro from "./components/DefaultOutro/DefaultOutro";
import DefaultIntro from "./components/DefaultIntro/DefaultIntro";
import LogoContainer from "@components/LogoContainer/LogoContainer";
import Button from "@components/Button/Button";

import { MultiStepSurveyProps } from "./MultiStepSurvey.props";
import { AnswerEntry } from "./MultiStepSurvey.types";

import classes from "./MultiStepSurvey.module.css";

const MultiStepSurvey: React.FC<MultiStepSurveyProps> = ({
  questions,
  Intro = DefaultIntro,
  Outro = DefaultOutro,
  userMeta = null,
}) => {
  const [currentStep, setCurrentStep] = useState<number>(-1);
  const [selectedAnswers, setSelectedAnswers] = useState<AnswerEntry[]>(
    questions.map((question) => ({
      question: question.text,
      answer: null,
    }))
  );

  const isLastStep = () => {
    return currentStep === questions.length - 1;
  };

  const AreAllQuestionsAnswered = () => {
    return selectedAnswers.every((entry) => entry.answer !== null);
  };

  const postSurveyResults = () => {
    const result = {
      ...(userMeta ? { userMeta } : {}),
      answers: selectedAnswers,
    };

    alert(JSON.stringify(result, null, 2));
  };

  if (currentStep === -1) {
    return <Intro onStepChange={setCurrentStep} />;
  } else if (currentStep === questions.length) {
    return <Outro />;
  } else {
    const currentQuestion = questions[currentStep];

    const backButtonClasses = classNames({
      [classes.HiddenButton]: currentStep <= 0,
    });
    const nextButtonClasses = classNames({
      [classes.HiddenButton]: isLastStep()
        ? !AreAllQuestionsAnswered()
        : !selectedAnswers[currentStep].answer,
    });

    const handleRadioClick = (answer: string) => {
      const updatedAnswers = selectedAnswers.map((entry, index) =>
        index === currentStep ? { ...entry, answer } : entry
      );

      setSelectedAnswers(updatedAnswers);
    };

    return (
      <LogoContainer>
        <section className={classes.MultiStepSurvey}>
          <ProgressNav
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
                  isSelected={selectedAnswers[currentStep].answer === answer}
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
