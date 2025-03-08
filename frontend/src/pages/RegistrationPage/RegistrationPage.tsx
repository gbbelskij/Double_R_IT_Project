import Logo from "@components/Logo/Logo";
import Main from "@components/Main/Main";

import "./RegistrationPage.css";

const RegistrationPage: React.FC = () => {
  return (
    <Main disableHeaderOffset>
      <Logo hasText />
    </Main>
  );
};

export default RegistrationPage;
